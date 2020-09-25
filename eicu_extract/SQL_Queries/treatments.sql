SELECT
  i.uniquePID                                     as subject_id,
  i.patientHealthSystemStayID                     as hadm_id,
  t.patientUnitStayID                             as icustay_id,
  ROUND(t.treatmentOffset::float / 60.0)::integer as hours_in,
  t.treatmentString                               as treatment_string -- This appears to be in a DSL of some kind.
FROM icustay_detail i
INNER JOIN treatment t ON i.patientUnitStayID = t.patientUnitStayID
WHERE t.patientUnitStayID IN ('{icustay_id}')
AND t.treatmentOffset > 0
AND t.treatmentOffset < i.unitDischargeOffset