
(SELECT e.itemid as ITEMID,
       f.label as LABEL,
       null as FLUID,
       f.category::VARCHAR as category,
       -- e.valueuom as UNITNAME,
       e.count as COUNT,
       e.min,
       e.max,
       f.LINKSTO,
       f.param_type,
       f.unitname

from

((select c.itemid,
       c.valueuom,
       count(c.value) as count,
       min(c.value)::VARCHAR as min,
       max(c.value)::VARCHAR as max --,
    --    'mimiciv_icu.chartevents' as LINKSTO
from mimiciv_icu.chartevents c
group by c.itemid, c.valueuom)


union

(select c.itemid,
       c.valueuom,
       count(c.value) as count,
       min(c.value)::VARCHAR as min,
       max(c.value)::VARCHAR as max --,
    --    'mimiciv_icu.outputevents' as LINKSTO
from mimiciv_icu.outputevents c
group by c.itemid, c.valueuom)

union

(select c.itemid,
       c.amountuom,
       count(c.amount) as count,
       min(c.amount)::VARCHAR as min,
       max(c.amount)::VARCHAR as max --,
    --    'mimiciv_icu.inputevents' as LINKSTO
from mimiciv_icu.inputevents c
group by c.itemid, c.amountuom)

union

(select c.itemid,
       c.valueuom,
       count(c.value) as count,
       min(c.value)::VARCHAR as min,
       max(c.value)::VARCHAR as max --,
    --    'mimiciv_icu.procedureevents' as LINKSTO
from mimiciv_icu.procedureevents c
group by c.itemid, c.valueuom)

union

(select c.itemid,
       c.valueuom,
       count(c.value) as count,
       min(c.value)::VARCHAR as min,
       max(c.value)::VARCHAR as max --,
    --    'mimiciv_icu.datetimeevents' as LINKSTO
from mimiciv_icu.datetimeevents c
group by c.itemid, c.valueuom)


) e
left join mimiciv_icu.d_items f
on e.itemid = f.itemid )

union



(SELECT e.itemid as ITEMID,
       d.label as LABEL,
       d.fluid as FLUID,
       d.category::VARCHAR as category,
       -- e.valueuom as UNITNAME,
       e.count as COUNT,
       e.min,
       e.max,
       e.LINKSTO,
       null as param_type,
       e.valueuom as unitname

from
(select c.itemid,
       c.valueuom,
       count(c.value) as count,
       min(c.value)::VARCHAR as min,
       max(c.value)::VARCHAR as max,
       'mimiciv_hosp.labevents' as LINKSTO
from mimiciv_hosp.labevents c
group by c.itemid, c.valueuom) e

left join mimiciv_hosp.d_labitems d
on e.itemid = d.itemid);


