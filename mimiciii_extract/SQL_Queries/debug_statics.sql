\echo "This file is just for debugging"
SET search_path TO mimiciii;
select distinct
    i.subject_id,
    i.hadm_id,
    i.icustay_id,
    i.gender,
    i.age as age,
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
--     c.timednr_chart,
    c.dnr_first_charttime,
    c.cmo_first,
    c.cmo_last,
    c.cmo,
    c.cmo_ds,
--     c.timecmo_chart,
    c.cmo_first_charttime,
--     c.timecmo_nursingnote,
    c.cmo_nursingnote_charttime,
    sofa.sofa,
    sofa.respiration as sofa_,
    sofa.coagulation as sofa_,
    sofa.liver as sofa_,
    sofa.cardiovascular as sofa_,
    sofa.cns as sofa_,
    sofa.renal as sofa_,
    sapsii.sapsii,
    sapsii.sapsii_prob,
    oasis.oasis,
    oasis.oasis_prob,
    COALESCE(f.readmission_30, 0) AS readmission_30
FROM icustay_detail i
    INNER JOIN admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN icustays s ON i.icustay_id = s.icustay_id
    INNER JOIN code_status c ON i.icustay_id = c.icustay_id
    LEFT OUTER JOIN (SELECT d.icustay_id, 1 as readmission_30
              FROM icustays c, icustays d
              WHERE c.subject_id=d.subject_id
              AND c.icustay_id > d.icustay_id
              AND c.intime - d.outtime <= interval '30 days'
              AND c.outtime = (SELECT MIN(e.outtime) from icustays e 
                                WHERE e.subject_id=c.subject_id
                                AND e.intime>d.outtime)) f
              ON i.icustay_id=f.icustay_id
    LEFT OUTER JOIN (SELECT icustay_id, sofa,  respiration, coagulation, liver, cardiovascular, cns, renal 
              FROM sofa) sofa
              ON i.icustay_id=sofa.icustay_id
    LEFT OUTER JOIN (SELECT icustay_id, sapsii,  sapsii_prob 
                FROM sapsii) sapsii
                ON sapsii.icustay_id=i.icustay_id
    LEFT OUTER JOIN (SELECT icustay_id, oasis, oasis_prob
                FROM oasis) oasis
                ON oasis.icustay_id=i.icustay_id
WHERE s.first_careunit NOT like 'NICU'
    and i.hadm_id is not null and i.icustay_id is not null
    and i.hospstay_seq = 1
    and i.icustay_seq = 1
    and i.age >= 16
    and i.los_icu >= 1
    and (i.outtime >= (i.intime + interval '12 hours'))
    and (i.outtime <= (i.intime + interval '250 hours'))
ORDER BY subject_id
