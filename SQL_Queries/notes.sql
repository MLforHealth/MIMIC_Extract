SELECT n.subject_id, n.hadm_id, i.icustay_id, n.chartdate, n.charttime, n.category, n.description, n.text
FROM noteevents n INNER JOIN icustays i on i.hadm_id = n.hadm_id
WHERE
  iserror IS NULL
  AND (n.chartdate <= i.outtime OR n.charttime <= i.outtime)
  AND n.hadm_id IN ('{hadm_id}')
  AND n.subject_id IN ('{subject_id}')
