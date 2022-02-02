--Dim tables help you decode SIDs of many types
select top 10 * from CDWWork.dim.LabChemTest




--in Src scheme, in the ORD database, these views have the "real" data. Relatively immutable.
--VINCI makes these for us
select top 10 
CohortName, LabChemSID, sta3n, LabChemTestSID, TopographySID, LOINCSID, LabChemResultNumericValue, LabChemResultValue, Units
-- , patientsid, labchemspecimendatetime
from ORD_Singh_201911038D.src.Chem_LabChem




--in ORD database, dflt schema, these tables have results that our team has stored.
-- we make these ourselves.
-- Four examples follow.
select count(*) from ORD_Singh_201911038D.[Dflt].[_A00_RFSPADETrgPos_Y201617_01_04_Cohort]  --306 patients
select count(*) from ORD_Singh_201911038D.[Dflt].[_A00_RFSPADETrgPos_Y201617_02_04_Demorgraphics]  --281
select count(*) from ORD_Singh_201911038D.[Dflt].[_A00_RFSPADETrgPos_Y201617_04_04_Lab]  --418784

select top 10
LabChemResultNumericValue, LabChemResultValue, labchemtestname, sta3n, LabChemTestSID, Abnormal, LOINC
from ORD_Singh_201911038D.[Dflt].[_A00_RFSPADETrgPos_Y201617_04_04_Lab]



-- One way to "dump the structured data"
-- may not be most efficient
--    select * from ORD_Singh_201911038D.[Dflt].[_A00_RFSPADETrgPos_Y201617_04_04_Lab]
-- use the export workflow (any one that works & preserves header row)
-- repeat 15 more times for all the other _A00_.... tables
-- some may be very very big.
-- this cohort is all patients who are trigger positive. From "refined SPADE trigger"




-- Inspecting the names of tables that Li already created.
select  * from ORD_Singh_201911038D.[INFORMATION_SCHEMA].TABLES
where TABLE_NAME like '_A00%'

select  
TABLE_CATALOG, TABLE_SCHEMA, 
[TABLE_NAME],
[COLUMN_NAME],
[ORDINAL_POSITION],
[DATA_TYPE],
[CHARACTER_MAXIMUM_LENGTH]
from ORD_Singh_201911038D.[INFORMATION_SCHEMA].COLUMNS
where TABLE_NAME like '_A00%'
order by TABLE_NAME, ORDINAL_POSITION



-- warning, real live query below.
select * from ORD_Singh_201911038D.[Dflt].[_A00_RFSPADETrgPos_Y201617_04_04_Lab]  -- 420k rows, 11 seconds. 41.4 MB
