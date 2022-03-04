 -- RadExam ( For patients with CTOrdered = 1 OR a.MRIOrdered=1)
  select ins.PatientSSN,rad.examdatetime,cptname,ins.[ER-PatientArrivalDateTime]
  ,ins.[ER-PatientDepartureDateTime],StudyID,RadiologyExamSID
  into dflt.[SPADE_Y20162017_TrgPos_RadExam]
from [ORD_Singh_201911038D].[Dflt].[SPADE_Y20162017_TrgPos_Correct] as ins
 left join src.Rad_RadiologyExam as rad
  on Rad.sta3n=ins.sta3n and rad.patientsid=ins.patientsid
  and ExamDateTime between [ER-PatientArrivalDateTime] and [ER-PatientDepartureDateTime]
  left join cdwwork.dim.[RadiologyProcedure] as prc
on rad.sta3n=prc.sta3n and rad.[RadiologyProcedureSID]=prc.[RadiologyProcedureSID]
left join cdwwork.dim.CPT as code
on prc.CPTSID=code.CPTSID and prc.sta3n=code.sta3n 
	where ins.patientssn in (
		select patientssn from [ORD_Singh_201911038D].Dflt._ForLi__INCLUSION_STEP02_v02 as a
		WHERE	(a.CTOrdered = 1 OR a.MRIOrdered = 1)
)

-- CPRSOrders ( For patients with CTOrdered = 1 OR a.MRIOrdered=1)
select ins.PatientSSN,rad.enteredDateTime,orderableItemname,ins.[ER-PatientArrivalDateTime]
  ,ins.[ER-PatientDepartureDateTime],StudyID,OrderedItemSID
  into dflt.[SPADE_Y20162017_TrgPos_CPRSOrder]
from [ORD_Singh_201911038D].[Dflt].[SPADE_Y20162017_TrgPos_Correct] as ins
left join [Src].CPRSOrder_OrderedItem as Rad
  on Rad.sta3n=ins.sta3n and rad.patientsid=ins.patientsid
    and EnteredDateTime between [ER-PatientArrivalDateTime] and [ER-PatientDepartureDateTime]
    left join cdwwork.dim.OrderableItem as ORI
  on ORI.OrderableItemSID=Rad.OrderableItemSID and ORI.sta3n=Rad.sta3n
	where ins.patientssn in (
		select  patientssn from [ORD_Singh_201911038D].Dflt._ForLi__INCLUSION_STEP02_v01 AS a
		WHERE (a.CTOrdered = 1 OR a.MRIOrdered = 1)
)
