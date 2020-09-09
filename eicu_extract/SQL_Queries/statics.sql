SELECT DISTINCT
    i.uniquePID                 as subject_id,
    i.patientHealthSystemStayID as hadm_id,
    i.patientUnitStayID         as icustay_id,
    i.hospitalid                as hospital_id,               -- unique to eICU
    i.region                    as region,                    -- unique to eICU
    i.unitVisitNumber           as icustay_seq,               -- Should always be 1!
    i.gender                    as gender,
    i.age                       as age,
    i.ethnicity                 as ethnicity,
    i.icu_los_hours / 24        as los_icu,                   -- Convert to days
    -- These serve as proxies for admission and discharge times
    i.hospitalAdmitOffset       as hospital_admit_offset,     -- unique to eICU
    i.hospitalDischargeOffset   as hospital_discharge_offset, -- unique to eICU
    i.unitAdmitOffset           as unit_admit_offset,         -- unique to eICU
    i.unitDischargeOffset       as unit_discharge_offset,     -- unique to eICU

    p.hospitalDischargeLocation as discharge_location,
    i.hosp_mort                 as mort_hosp,
    CASE WHEN LOWER(p.unitDischargeStatus) LIKE '%alive%' THEN 0
         WHEN LOWER(p.unitDischargeStatus) LIKE '%expired%' THEN 1
         ELSE NULL END AS mort_icu,


    i.hospitalDischargeYear     as hospital_discharge_year,   -- unique to eICU
    i.unitType                  as unit_type,                 -- unique to eICU; maybe first_careunit analog?

    -- Missing (compared to mimic_extract)
    -- TODO: IMPORTANT: hospstay_seq. We limit this to 1 in MIMIC extract, but don't have it here.
    -- admissiondx.admitDxName          as diagnosis_at_admission,
    -- admissiondx.admitDxEnteredOffset as admission_diagnosis_offset_minutes,
    -- a.admission_type,
    -- a.insurance, -- NOT IN eICU
    -- a.deathtime, -- Double duty with hospitalDischargeOffset in case that hospital_expire_flag is true.
    -- i.hospital_expire_flag, -- this was redundant anyways
    -- s.first_careunit,
    -- c.fullcode_first,
    -- c.dnr_first,
    -- c.fullcode,
    -- c.dnr,
    -- c.dnr_first_charttime,
    -- c.cmo_first,
    -- c.cmo_last,
    -- c.cmo,
    -- c.timecmo_chart,
    -- sofa.sofa,
    -- sofa.respiration as sofa_,
    -- sofa.coagulation as sofa_,
    -- sofa.liver as sofa_,
    -- sofa.cardiovascular as sofa_,
    -- sofa.cns as sofa_,
    -- sofa.renal as sofa_,
    -- sapsii.sapsii,
    -- sapsii.sapsii_prob,
    -- oasis.oasis,
    -- oasis.oasis_prob,
    -- readmission_30,

    -- Missing (but could be pulled from eICU concepts)
    -- i.admissionHeight
    -- i.admissionWeight
    -- p.unitAdmitTime24
    -- p.unitStayType


FROM icustay_detail i
  INNER JOIN patients p ON i.uniquePID = p.uniquePID AND i.patientUnitStayID = p.patientUnitStayID
WHERE i.patientHealthSystemStayID is not null and i.patientUnitStayID is not null
    and i.unitVisitNumber = 1
    and i.age >= {min_age}
    and i.icu_los_hours / 24 >= {min_day}
    and (i.unitDischargeOffset - i.unitAdmitOffset) / 60 >= {min_dur}
    and (i.unitDischargeOffset - i.unitAdmitOffset) / 60 <= {max_dur}
ORDER BY subject_id
{limit}
