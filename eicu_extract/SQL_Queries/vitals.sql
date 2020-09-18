SELECT
  i.uniquePID                 as subject_id,
  i.patientHealthSystemStayID as hadm_id,
  CASE WHEN vp.patientUnitStayID IS NOT NULL THEN vp.patientUnitStayID
       WHEN vap.patientUnitStayID IS NOT NULL THEN vap.patientUnitStayID 
       ELSE NULL END as icustay_id,
  CASE WHEN vp.observationOffset IS NOT NULL THEN ROUND(vp.observationOffset::float / 60.0)::integer
       WHEN vap.observationOffset IS NOT NULL THEN ROUND(vap.observationOffset::float / 60.0)::integer
       ELSE NULL END as hours_in,
  vp.temperature              as temperature,
  vp.saO2                     as SaO2,
  vp.heartRate                as heart_rate,
  vp.respiration              as respiratory_rate,       -- TODO(mmd): is this right?
  vp.cvp                      as cvp,
  vp.etCo2                    as etCO2,
  vp.systemicSystolic         as systemic_systolic,
  vp.systemicDiastolic        as systemic_diastolic,
  vp.systemicMean             as systemic_mean,
  vp.paSystolic               as pa_systolic,
  vp.paDiastolic              as pa_diastolic,
  vp.paMean                   as pa_mean,
  vp.st1                      as st1,
  vp.st2                      as st2,
  vp.st3                      as st3,
  vp.ICP                      as ICP,
  vap.nonInvasiveSystolic     as noninvasive_systolic,
  vap.nonInvasiveDiastolic    as noninvasive_diastolic,
  vap.nonInvasiveMean         as noninvasive_mean,
  vap.paop                    as paop,
  vap.cardiacOutput           as cardiac_output,
  vap.cardiacInput            as cardiac_input,
  vap.svr                     as svr,
  vap.svri                    as svri,
  vap.pvr                     as pvr,
  vap.pvri                    as pvri
FROM (
    vitalPeriodic vp FULL OUTER JOIN vitalAperiodic vap ON
      vp.patientUnitStayID = vap.patientUnitStayID
      AND vp.observationOffset = vap.observationOffset
)
INNER JOIN icustay_detail i ON i.patientUnitStayID = vap.patientUnitStayID
WHERE i.patientUnitStayID IN ('{icustay_id}')