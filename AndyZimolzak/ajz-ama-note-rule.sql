select top 20 * FROM [ORD_Singh_201911038D].[Src].[TIU_TIUDocument]

SELECT TOP (5) [CohortName]
      ,[TIUDocumentSID]
      ,[TIUDocumentIEN]
      ,[Sta3n]
      ,[TIUDocumentDefinitionSID]
      ,[ReferenceDateTime]
      ,[SignatureDateTime]
      ,[CosignatureDateTime]
      ,[AmendmentDateTime]
      ,[ProcedureDateTime]
FROM [ORD_Singh_201911038D].[Src].[TIU_TIUDocument]
where TIUDocumentDefinitionSID = 1000151218

select top 30 * from cdwwork.[Dim].[TIUDocumentDefinition] where tiudocumentdefinition like 'leav%' 

select TIUDocumentDefinitionSID, sta3n, TIUDocumentDefinition
 from cdwwork.[Dim].[TIUDocumentDefinition] 
	where (tiudocumentdefinition like '%imed%' or TIUDocumentDefinition like '%i-med%') and (TIUDocumentDefinition like '%ama%' or TIUDocumentDefinition like '%against%')
	order by sta3n
	--148 rows

-- for 580, sid = 1000151218

select top 30 * from cdwwork.[Dim].[TIUDocumentType] where (TIUDocumentType like '%against%' or TIUDocumentType like '%ama%') and sta3n=580 --against medical advice note, ama note
-- not useful


select TIUDocumentDefinitionSID, sta3n, TIUDocumentDefinition
 from cdwwork.[Dim].[TIUDocumentDefinition] 
	where  (TIUDocumentDefinition like '%ama%' or TIUDocumentDefinition like '%against%') 
	and TIUDocumentDefinition not like '%carbam%' and TIUDocumentDefinition not like '%tramad%'  and TIUDocumentDefinition not like '%amarillo%'  and TIUDocumentDefinition not like '%amat%'  and TIUDocumentDefinition not like '%lamar%' 
	 and TIUDocumentDefinition not like '%amax%' 
	order by sta3n