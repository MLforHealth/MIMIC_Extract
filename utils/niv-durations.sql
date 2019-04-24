-- This query extracts the duration of mechanical ventilation
-- The main goal of the query is to aggregate sequential ventilator settings
-- into single mechanical ventilation "events". The start and end time of these
-- events can then be used for various purposes: calculating the total duration
-- of mechanical ventilation, cross-checking values (e.g. PaO2:FiO2 on vent), etc

-- The query's logic is roughly:
--    1) The presence of a mechanical ventilation setting starts a new ventilation event
--    2) Any instance of a setting in the next 8 hours continues the event
--    3) Certain elements end the current ventilation event
--        a) documented extubation ends the current ventilation
--        b) initiation of non-invasive vent and/or oxygen ends the current vent
-- The ventilation events are numbered consecutively by the `num` column.


-- First, create a temporary table to store relevant data from CHARTEVENTS.
DROP  MATERIALIZED VIEW IF EXISTS nivdurations CASCADE;
create  MATERIALIZED VIEW nivdurations as
with nivsettings AS
(
select
  icustay_id, charttime
    , max(
      case
        -- initiation of oxygen therapy
        when itemid = 226732 and value in
        (
          'Nasal cannula', -- 153714 observations
          'Face tent', -- 24601 observations
          'Aerosol-cool', -- 24560 observations
          'Trach mask ', -- 16435 observations
          'High flow neb', -- 10785 observations
          'Non-rebreather', -- 5182 observations
          'Venti mask ', -- 1947 observations
          'Medium conc mask ', -- 1888 observations
          'T-piece', -- 1135 observations
          'High flow nasal cannula', -- 925 observations
          'Ultrasonic neb', -- 9 observations
          'Vapomist' -- 3 observations
        ) then 1
        when itemid in (467,468) and value in
        (
          'Cannula', -- 278252 observations
          'Nasal Cannula', -- 248299 observations
          'None', -- 95498 observations
          'Face Tent', -- 35766 observations
          'Aerosol-Cool', -- 33919 observations
          'Trach Mask', -- 32655 observations
          'Hi Flow Neb', -- 14070 observations
          'Non-Rebreather', -- 10856 observations
          'Venti Mask', -- 4279 observations
          'Medium Conc Mask', -- 2114 observations
          'Vapotherm', -- 1655 observations
          'T-Piece', -- 779 observations
          'Hood', -- 670 observations
          'Hut', -- 150 observations
          'TranstrachealCat', -- 78 observations
          'Heated Neb', -- 37 observations
          'Ultrasonic Neb' -- 2 observations
        ) then 1
      when itemid = 469 and value in ('Nasal Cannula', 'Face Tent', 'Trach Mask') then 1
      when itemid in (470, 471, 227287, 223834) and valuenum > 0 then 1
      else 0
      end
    ) as OxygenTherapy
from chartevents ce
where ce.value is not null
-- exclude rows marked as error
and ce.error IS DISTINCT FROM 1
and itemid in
(
    -- the below indicate oxygen/NIV
      467 -- O2 Delivery Device
    , 468 -- O2 Delivery Device#2
    , 469 -- O2 Delivery Mode
    , 470 -- O2 Flow (lpm)
    , 471 -- O2 Flow (lpm) #2
    , 227287 -- O2 Flow (additional cannula)
    , 226732 -- O2 Delivery Device(s)
    , 223834 -- O2 Flow
)
group by icustay_id, charttime
)
, vd0 as
(
  select
    icustay_id
    -- this carries over the previous charttime which had a mechanical ventilation event
    , case
        when OxygenTherapy=1 then
          LAG(CHARTTIME, 1) OVER (partition by icustay_id, OxygenTherapy order by charttime)
        else null
      end as charttime_lag
    , charttime
    , OxygenTherapy
  from nivsettings
)
, vd1 as
(
  select
      icustay_id
      , charttime_lag
      , charttime
      , OxygenTherapy

      -- if this is a mechanical ventilation event, we calculate the time since the last event
      , case
          -- if the current observation indicates mechanical ventilation is present
          -- calculate the time since the last vent event
          when OxygenTherapy=1 then
            CHARTTIME - charttime_lag
          else null
        end as ventduration

      , case when (CHARTTIME - charttime_lag) > interval '8' hour then 1
        else 0
        end as newvent
  -- use the staging table with only vent settings from chart events
  FROM vd0
)
, vd2 as
(
  select vd1.*
  -- create a cumulative sum of the instances of new ventilation
  -- this results in a monotonic integer assigned to each instance of ventilation
  , case when OxygenTherapy=1 then
      SUM( newvent )
      OVER ( partition by icustay_id order by charttime )
    else null end
    as ventnum
  --- now we convert CHARTTIME of ventilator settings into durations
  from vd1
)
-- create the durations for each mechanical ventilation instance
select icustay_id
  -- regenerate ventnum so it's sequential
  , ROW_NUMBER() over (partition by icustay_id order by ventnum) as ventnum
  , min(charttime) as starttime
  , max(charttime) as endtime
  , extract(epoch from max(charttime)-min(charttime))/60/60 AS duration_hours
from vd2
group by icustay_id, ventnum
having min(charttime) != max(charttime)
-- patient had to be given NIV at least once
-- i.e. max(OxygenTherapy) should be 1
and max(OxygenTherapy) = 1
order by icustay_id, ventnum;
