/*********** creat **********/


select count(*) from  [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
-- 2447196

SELECT TOP 300 * -- LabChemResultNumericValue, LabChemTestName, LOINC
  FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
 
  
  where LabChemTestName like 'creatini%'
  and not loinc = '2161-8' -- known BAD, urine!
  and not loinc = '2160-0' --known good loinc

  /*
  maybe okay loincs in addition to the popular 2160-0

  21232-4
  38483-4 = "istat"

  */


SELECT TOP 300 *
FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
 where (
 loinc = '2160-0' or
 loinc = '21232-4' or
 loinc = '38483-4'
 )



 /********** wbc ********/

 SELECT TOP 300 *
  FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
  where LabChemTestName like 'wbc'
  -- best loinc 6690-2

SELECT loinc, LabChemTestName, count(loinc) as n
  FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
  where LabChemTestName like 'wbc'
  group by loinc, LabChemTestName
  order by n desc


  /*
  loinc	LabChemTestName
805-2	WBC
804-5	WBC
814-4	WBC
26464-8	WBC
806-0	WBC
26467-1	WBC
51383-8	WBC
6690-2	WBC
14810-6	WBC
33256-9	WBC
6743-9	WBC
26469-7	WBC
*Missing*	WBC
808-6	WBC
813-6	WBC
26468-9	WBC
810-2	WBC
*/


 SELECT TOP 300 *
  FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
  where LabChemTestName like 'wbc'
  and not loinc = '6690-2'
  -- some of these not popular loincs have strange high values like 396, not real blood WBC counts.






  /************** sodium **********/

   SELECT top 300 *
  FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
  where LabChemTestName like 'sodium%'
  
-- best one loinc is 2951-2
-- also good 32717-1

SELECT loinc, LabChemTestName, count(loinc) as n
  FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
  where LabChemTestName like 'sodium%'
  group by loinc, LabChemTestName
  order by loinc

  --3rd best, mysterious: 2947-0 --> this is ok to include. POC or blood gas.
  -- last one okay to includ: 39791-9
 -- altogether: 4 loincs to include.


  -- ok to exclude below this line
    -- mysterious but n=3 : 27419-1
  -- mystery: 2950-4 





  /************ calcium ********/

SELECT loinc, LabChemTestName, count(loinc) as n
  FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
  where LabChemTestName like 'calcium%'
    group by loinc, LabChemTestName
  order by loinc
  -- good ones: 
  -- 17861-6  2000-8  


   SELECT top 300 *
  FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
  where loinc = '2000-8'



  /******* lactate */

SELECT top 300 *
  FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
  where LabChemTestName like 'lact%'

  SELECT loinc, LabChemTestName, count(loinc) as n
  FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
  where LabChemTestName like 'lact%'
    group by loinc, LabChemTestName
  order by loinc

  /*
  best:
  2524-7

probably good loincs:
  14118-4
  19240-1
2518-9
2519-7
2520-5
30241-4
32693-4

  */




  /******** bun
  "UREA NITROGEN, BLOOD"
*/


SELECT top 300 *
  FROM [ORD_Singh_201911038D].[Dflt].[_B00_ML4TrgPos_Y201621_04_04_Lab]
  where LabChemTestName like 'lact%'