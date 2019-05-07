select distinct
    i.subject_id,
    i.hadm_id,
    i.icustay_id,
    i.gender,
    i.admission_age as age,
    i.ethnicity,
    i.admission_type,
    i.hospital_expire_flag,
    i.hospstay_seq,
    i.los_icu,
    i.admittime,
    i.dischtime,
    i.intime,
    i.outtime,
    a.diagnosis AS diagnosis_at_admission,
    a.insurance,
    a.deathtime,
    a.discharge_location,
    CASE when a.deathtime between i.intime and i.outtime THEN 1 ELSE 0 END AS mort_icu,
    CASE when a.deathtime between i.admittime and i.dischtime THEN 1 ELSE 0 END AS mort_hosp,
    s.first_careunit,
    c.fullcode_first,
    c.dnr_first,
    c.fullcode,
    c.dnr,
    c.timednr_chart,
    c.cmo_first,
    c.cmo_last,
    c.cmo,
    c.cmo_ds,
    c.timecmo_chart,
    c.timecmo_nursingnote
FROM icustay_detail i
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN icustays s ON i.icustay_id = s.icustay_id
    INNER JOIN code_status c ON i.icustay_id = c.icustay_id
WHERE s.first_careunit NOT like 'NICU'
    and i.hadm_id is not null and i.icustay_id is not null
    and i.hospstay_seq = 1
    and i.icustay_seq = 1
    and i.admission_age >= {min_age}
    and i.los_icu >= {min_day}
    and (i.outtime >= (i.intime + interval '{min_dur} hours'))
    and (i.outtime <= (i.intime + interval '{max_dur} hours'))
ORDER BY subject_id
{limit}
