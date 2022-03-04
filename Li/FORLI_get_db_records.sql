USE ORD_Singh_201911038D


-- STEP 01 - Connects the patient records in the SPADE ML Access Chart
-- ************************************************************************************************


-- STEP 01v01
-- ------------------------------------------------------------------------------------------------


IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP01_v01') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP01_v01
	END

CREATE TABLE Dflt._ForLi__INCLUSION_STEP01_v01
(
	StudyID VARCHAR(50) DEFAULT NULL
	,PatientSSN_rec VARCHAR(50) DEFAULT NULL
	,PatientSSN_db VARCHAR(50) DEFAULT NULL
	,PatientSID BIGINT DEFAULT -1
	,EDSID BIGINT DEFAULT -1
	,EDStartDateTime_rec DATE DEFAULT NULL
	,EDStartDateTime_db DATETIME2 DEFAULT NULL
	,EDStopDateTime DATETIME2 DEFAULT NULL
	,InpSID BIGINT DEFAULT -1
	,InpStartDateTime_rec DATE DEFAULT NULL
	,InpStartDateTime_db DATETIME2 DEFAULT NULL
	,InpStopDateTime DATETIME2 DEFAULT NULL
)


INSERT INTO Dflt._ForLi__INCLUSION_STEP01_v01
SELECT DISTINCT
	map.[Study ID] AS StudyID
	,map.PatientSSN AS PatientSSN_rec
	,sp.PatientSSN AS PatientSSN_db
	,sp.PatientSID
	,edis.VisitSID
	,CONVERT(DATE, map.[ER-PatientArrivalDateTime]) AS EDStartDateTime_rec
	,edis.PatientArrivalDateTime AS EDStartDateTime_db
	,edis.PatientDepartureDateTime
	,inp.InpatientSID
	,CONVERT(DATE, map.AdmitDateTime) AS InpStartDateTime_rec
	,inp.AdmitDateTime AS InpStartDateTime_db
	,inp.DischargeDateTime
FROM
	Dflt.SPADE_StudyIDMapping_Table2 AS map INNER JOIN Dflt.SPADE_Y20162017_TrgPos_Correct AS results
		ON map.PatientSSN = results.PatientSSN
			INNER JOIN Dflt.SPADE_ChartReview_Table AS charts
				ON charts.StudyID = map.[Study ID]
					LEFT JOIN Src.EDIS_EDISLog AS edis
						ON edis.VisitSID = results.ERVisitSID
							LEFT JOIN Src.SPatient_SPatient AS sp ON
								sp.PatientSID = edis.PatientSID
									LEFT JOIN Src.Inpat_Inpatient AS inp ON
										inp.InpatientSID = results.inpatientsid
WHERE
	CONVERT(DATE, inp.AdmitDateTime) = CONVERT(DATE, map.AdmitDateTime)
	AND
	CONVERT(DATE, map.[ER-PatientArrivalDateTime]) = CONVERT(DATE, edis.PatientArrivalDateTime)
ORDER BY StudyID ASC


-- STEP 01v02
-- ------------------------------------------------------------------------------------------------


IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP01_v02') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP01_v02
	END

CREATE TABLE Dflt._ForLi__INCLUSION_STEP01_v02
(
	StudyID VARCHAR(50) DEFAULT NULL
	,PatientSSN_rec VARCHAR(50) DEFAULT NULL
	,PatientSSN_db VARCHAR(50) DEFAULT NULL
	,PatientSID BIGINT DEFAULT -1
	,EDSID BIGINT DEFAULT -1
	,EDStartDateTime_rec DATE DEFAULT NULL
	,EDStartDateTime_db DATETIME2 DEFAULT NULL
	,EDStopDateTime DATETIME2 DEFAULT NULL
	,InpSID BIGINT DEFAULT -1
	,InpStartDateTime_rec DATE DEFAULT NULL
	,InpStartDateTime_db DATETIME2 DEFAULT NULL
	,InpStopDateTime DATETIME2 DEFAULT NULL
)


INSERT INTO Dflt._ForLi__INCLUSION_STEP01_v02
SELECT DISTINCT
	*
FROM Dflt._ForLi__INCLUSION_STEP01_v01 AS prev
WHERE
	prev.EDStartDateTime_db = (SELECT MIN(t.EDStartDateTime_db) FROM Dflt._ForLi__INCLUSION_STEP01_v01 AS t WHERE t.StudyID = prev.StudyID)
	AND
	prev.InpStartDateTime_db = (SELECT MIN(t.InpStartDateTime_db) FROM Dflt._ForLi__INCLUSION_STEP01_v01 AS t WHERE t.StudyID = prev.StudyID)


-- STEP 01vFIN
-- ------------------------------------------------------------------------------------------------


IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP01_vFIN') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP01_vFIN
	END

CREATE TABLE Dflt._ForLi__INCLUSION_STEP01_vFIN
(
	StudyID VARCHAR(50) DEFAULT NULL
	,PatientSSN VARCHAR(50) DEFAULT NULL
	,PatientSID BIGINT DEFAULT -1
	,EDSID BIGINT DEFAULT -1
	,EDStartDateTime DATETIME2 DEFAULT NULL
	,EDStopDateTime DATETIME2 DEFAULT NULL
	,InpSID BIGINT DEFAULT -1
	,InpStartDateTime DATETIME2 DEFAULT NULL
	,InpStopDateTime DATETIME2 DEFAULT NULL
)


INSERT INTO Dflt._ForLi__INCLUSION_STEP01_vFIN
SELECT DISTINCT
	prev.StudyID
	,prev.PatientSSN_db
	,prev.PatientSID
	,prev.EDSID
	,prev.EDStartDateTime_db
	,prev.EDStopDateTime
	,prev.InpSID
	,prev.InpStartDateTime_db
	,prev.InpStopDateTime
FROM
	Dflt._ForLi__INCLUSION_STEP01_v02 AS prev


-- STEP 02 - Connects to CPRSOrders
-- ************************************************************************************************


IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP02_v01') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP02_v01
	END

CREATE TABLE Dflt._ForLi__INCLUSION_STEP02_v01
(
	StudyID VARCHAR(50) DEFAULT NULL
	,PatientSSN VARCHAR(50) DEFAULT NULL
	,CTOrdered INT DEFAULT -1
	,MRIOrdered INT DEFAULT -1
	,Item VARCHAR(100) DEFAULT NULL
)


INSERT INTO Dflt._ForLi__INCLUSION_STEP02_v01
SELECT
	prev.StudyID
	,prev.PatientSSN
	,CONVERT(INT, charts.CTatER)
	,CONVERT(INT, charts.MRIatER)
	,ordbi.OrderableItemName
FROM
	Dflt._ForLi__INCLUSION_STEP01_vFIN AS prev INNER JOIN Dflt.SPADE_ChartReview_Table AS charts ON
	(
		charts.StudyID = prev.StudyID
	)
		LEFT JOIN Src.CPRSOrder_CPRSOrder AS ord ON
		(
			ord.PatientSID = prev.PatientSID
			AND
			ord.OrderStartDateTime BETWEEN
				prev.EDStartDateTime
				AND
				DATEADD(HOUR, 6, prev.EDStopDateTime)
		)
			LEFT JOIN Src.CPRSOrder_OrderedItem AS ordi ON
			(
				ord.CPRSOrderSID = ordi.CPRSOrderSID
			)
				LEFT JOIN CDWWork.Dim.OrderableItem AS ordbi ON
				(
					ordi.OrderableItemSID = ordbi.OrderableItemSID
				)
ORDER BY
	prev.StudyID



IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP02_v02') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP02_v02
	END

CREATE TABLE Dflt._ForLi__INCLUSION_STEP02_v02
(
	StudyID VARCHAR(50) DEFAULT NULL
	,PatientSSN VARCHAR(50) DEFAULT NULL
	,CTOrdered INT DEFAULT -1
	,MRIOrdered INT DEFAULT -1
	,Item VARCHAR(100) DEFAULT NULL
)


INSERT INTO Dflt._ForLi__INCLUSION_STEP02_v02
SELECT
	prev.StudyID
	,prev.PatientSSN
	,CONVERT(INT, charts.CTatER)
	,CONVERT(INT, charts.MRIatER)
	,rp.RadiologyProcedure
FROM
	Dflt._ForLi__INCLUSION_STEP01_vFIN AS prev INNER JOIN Dflt.SPADE_ChartReview_Table AS charts ON
	(
		charts.StudyID = prev.StudyID
	)
		LEFT JOIN Src.Rad_RadiologyExam AS rad ON
		(
			rad.PatientSID = prev.PatientSID
			AND 
			rad.RequestedDateTime BETWEEN
				prev.EDStartDateTime
				AND
				DATEADD(HOUR, 6, prev.EDStopDateTime)
		)
			LEFT JOIN CDWWork.Dim.RadiologyProcedure AS rp ON
			(
				rad.RadiologyProcedureSID = rp.RadiologyProcedureSID
			)
ORDER BY
	prev.StudyID


-- STEP 02vFIN
-- ------------------------------------------------------------------------------------------------


IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP02_vFIN') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP02_vFIN
	END

CREATE TABLE Dflt._ForLi__INCLUSION_STEP02_vFIN
(
	StudyID VARCHAR(50) DEFAULT NULL
	,PatientSSN VARCHAR(50) DEFAULT NULL
	,PatientSID BIGINT DEFAULT -1
	,EDSID BIGINT DEFAULT -1
	,EDStartDateTime DATETIME2 DEFAULT NULL
	,EDStopDateTime DATETIME2 DEFAULT NULL
	,InpSID BIGINT DEFAULT -1
	,InpStartDateTime DATETIME2 DEFAULT NULL
	,InpStopDateTime DATETIME2 DEFAULT NULL
)


INSERT INTO Dflt._ForLi__INCLUSION_STEP02_vFIN
SELECT *
FROM Dflt._ForLi__INCLUSION_STEP01_vFIN AS prev
WHERE
	prev.StudyID IN
	(
		SELECT z.StudyID
		FROM Dflt._ForLi__INCLUSION_STEP02_v02 AS z
		WHERE
			z.Item LIKE '%CT%HEAD%'
			OR z.Item LIKE '%CTA%HEAD%'
			OR z.Item LIKE '%CT%BRAIN%'
			OR z.Item LIKE '%CTA%BRAIN%'
			OR z.Item LIKE 'HEAD W/0 CONT (CT)'
			OR z.Item LIKE 'SPINE- CERVICAL W/O CONT (CT)'
			OR z.Item LIKE '%MRI%BRAIN%'
			OR z.Item LIKE '%MRI%HEAD%'
			OR z.Item LIKE '%MRI%NECK%'
			OR z.Item LIKE '%MRA%BRAIN%'
			OR z.Item LIKE '%MRA%HEAD%'
			OR z.Item LIKE '%MRA%NECK%'
			OR z.Item LIKE 'MAGNETIC IMAGE,BRAIN W/&W/O CONTRAST'
	)


-- PRINT
-- ************************************************************************************************
-- Patients/StudyIDs that were found to have MRIs or CTs in the ED during manual review but have no such record in radiology domain
SELECT *
FROM Dflt._ForLi__INCLUSION_STEP02_v02 AS a
WHERE
	(a.CTOrdered = 1 OR a.MRIOrdered = 1)
	AND
	a.StudyID NOT IN
	(
		SELECT z.StudyID
		FROM Dflt._ForLi__INCLUSION_STEP02_v02 AS z
		WHERE
			z.Item LIKE '%CT%HEAD%'
			OR z.Item LIKE '%CTA%HEAD%'
			OR z.Item LIKE '%CT%BRAIN%'
			OR z.Item LIKE '%CTA%BRAIN%'
			OR z.Item LIKE 'HEAD W/0 CONT (CT)'
			OR z.Item LIKE 'SPINE- CERVICAL W/O CONT (CT)'
			OR z.Item LIKE '%MRI%BRAIN%'
			OR z.Item LIKE '%MRI%HEAD%'
			OR z.Item LIKE '%MRI%NECK%'
			OR z.Item LIKE '%MRA%BRAIN%'
			OR z.Item LIKE '%MRA%HEAD%'
			OR z.Item LIKE '%MRA%NECK%'
			OR z.Item LIKE 'MAGNETIC IMAGE,BRAIN W/&W/O CONTRAST'
	)
ORDER BY a.StudyID ASC


-- Patients/StudyIDs that were found to have MRIs or CTs in the ED during manual review but have no such record in CPRSOrder domain
SELECT *
FROM Dflt._ForLi__INCLUSION_STEP02_v01 AS a
WHERE
	(a.CTOrdered = 1 OR a.MRIOrdered = 1)
	AND
	a.StudyID NOT IN
	(
		SELECT z.StudyID
		FROM Dflt._ForLi__INCLUSION_STEP02_v01 AS z
		WHERE
			z.Item LIKE '%CT%HEAD%'
			OR z.Item LIKE '%CTA%HEAD%'
			OR z.Item LIKE '%CT%BRAIN%'
			OR z.Item LIKE '%CTA%BRAIN%'
			OR z.Item LIKE 'HEAD W/0 CONT (CT)'
			OR z.Item LIKE 'SPINE- CERVICAL W/O CONT (CT)'
			OR z.Item LIKE '%MRI%BRAIN%'
			OR z.Item LIKE '%MRI%HEAD%'
			OR z.Item LIKE '%MRI%NECK%'
			OR z.Item LIKE '%MRA%BRAIN%'
			OR z.Item LIKE '%MRA%HEAD%'
			OR z.Item LIKE '%MRA%NECK%'
			OR z.Item LIKE 'MAGNETIC IMAGE,BRAIN W/&W/O CONTRAST'
	)
ORDER BY a.StudyID ASC


-- CLEAN UP
-- ************************************************************************************************


IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP01_v01') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP01_v01
	END

IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP01_v02') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP01_v02
	END

IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP01_vFIN') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP01_vFIN
	END

IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP02_v01') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP02_v01
	END

IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP02_v02') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP02_v02
	END

IF (OBJECT_ID('Dflt._ForLi__INCLUSION_STEP02_vFIN') IS NOT NULL)
	BEGIN
		DROP TABLE Dflt._ForLi__INCLUSION_STEP02_vFIN
	END