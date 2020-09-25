SELECT
  i.uniquePID                                     as subject_id,
  i.patientHealthSystemStayID                     as hadm_id,
  l.patientUnitStayID                             as icustay_id,
  ROUND(l.labResultOffset::float / 60.0)::integer as hours_in,       -- matching the naming schema of MIMIC for now...
  l.labName                                       as itemid, -- matching the naming schema of MIMIC for now, but this may be super wrong -- there may be big confounders here across hospitals.
  l.labResult                                     as value,
  l.labMeasureNameSystem                          as valueuom_system,
  l.labMeasureNameInterface                       as valueuom_interface
FROM icustay_detail i
INNER JOIN lab l ON i.patientUnitStayID = l.patientUnitStayID
where l.patientUnitStayID IN ('{icustay_id}')
  AND l.labResult IS NOT NULL
  AND l.labResultOffset > 0 -- Only take labs during the unit stay.
  AND l.labResultOffset < i.unitDischargeOffset  -- Only take labs during the unit stay.