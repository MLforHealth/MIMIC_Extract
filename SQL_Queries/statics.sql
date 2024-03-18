select distinct
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.gender,
    i.admission_age as age,
    i.race,
    i.hospital_expire_flag,
    i.hospstay_seq,
    i.los_icu,
    i.admittime,
    i.dischtime,
    i.icu_intime as intime,
    i.icu_outtime as outtime,
    --a.diagnosis AS diagnosis_at_admission,
    a.admission_type,
    a.insurance,
    a.deathtime,
    a.discharge_location,
    CASE when a.deathtime between i.icu_intime and i.icu_outtime THEN 1 ELSE 0 END AS mort_icu,
    CASE when a.deathtime between i.admittime and i.dischtime THEN 1 ELSE 0 END AS mort_hosp,
    s.first_careunit,
    --c.fullcode_first,
    --c.dnr_first,
    --c.fullcode,
    --c.dnr,
    --c.dnr_first_charttime,
    --c.cmo_first,
    --c.cmo_last,
    --c.cmo,
    --c.timecmo_chart,
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
FROM mimiciv_derived.icustay_detail i
    INNER JOIN mimiciv_hosp.admissions a ON i.hadm_id = a.hadm_id
    INNER JOIN mimiciv_icu.icustays s ON i.stay_id = s.stay_id
    --INNER JOIN code_status c ON i.hadm_id = c.hadm_id
    LEFT OUTER JOIN (SELECT d.stay_id, 1 as readmission_30
              FROM icustays c, icustays d
              WHERE c.subject_id=d.subject_id
              AND c.stay_id > d.stay_id
              AND c.intime - d.outtime <= interval '30 days'
              AND c.outtime = (SELECT MIN(e.outtime) from icustays e 
                                WHERE e.subject_id=c.subject_id
                                AND e.intime>d.outtime)) f
              ON i.stay_id=f.stay_id
    LEFT OUTER JOIN (SELECT stay_id, max(sofa_24hours) AS sofa,  max(respiration) as respiration, min(coagulation) as coagulation, max(liver) as liver, max(cardiovascular) as cardiovascular, max(cns) as cns, max(renal) as renal 
              FROM mimiciv_derived.sofa group by stay_id) sofa
              ON i.stay_id=sofa.stay_id
    LEFT OUTER JOIN (SELECT stay_id, sapsii,  sapsii_prob 
                FROM mimiciv_derived.sapsii) sapsii
                ON sapsii.stay_id=i.stay_id
    LEFT OUTER JOIN (SELECT stay_id, oasis, oasis_prob
                FROM mimiciv_derived.oasis) oasis
                ON oasis.stay_id=i.stay_id
WHERE s.first_careunit NOT like 'NICU'
    and i.hadm_id is not null and i.stay_id is not null
    and i.hospstay_seq = 1
    and i.icustay_seq = 1
    and i.admission_age >= {min_age}
    and i.los_icu >= {min_day}
    and (i.icu_outtime >= (i.icu_intime + interval '{min_dur} hours'))
    and (i.icu_outtime <= (i.icu_intime + interval '{max_dur} hours'))
ORDER BY subject_id
{limit}
