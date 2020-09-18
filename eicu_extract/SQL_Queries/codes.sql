SELECT
  i.uniquePID                 as subject_id,
  i.patientHealthSystemStayID as hadm_id,
  d.patientUnitStayID         as icustay_id,
  --d.diagnosisOffset         as observation_offset_min, -- omitted as this is actually offset of entry, not of diagnosis.
  d.diagnosisString           as diagnosisString,
  d.ICD9Code                  as icd9_code,
  d.diagnosisPriority         as diagnosisPriority
FROM icustay_detail i
INNER JOIN diagnosis d ON i.patientUnitStayID = d.patientUnitStayID
where d.patientUnitStayID IN ({icuids})