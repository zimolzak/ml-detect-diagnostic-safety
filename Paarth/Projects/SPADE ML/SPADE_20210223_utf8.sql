﻿
if (OBJECT_ID('[Dflt].[SpadeRF_00_1_inputP]') is not null)	
	drop table [Dflt].SpadeRF_00_1_inputP
	
		CREATE TABLE [Dflt].SpadeRF_00_1_inputP(
		[sp_start] datetime2(0) NULL,
		[sp_end] datetime2(0) NULL)
	
INSERT INTO [Dflt].[SpadeRF_00_1_inputP]
           (
           [sp_start]
           ,[sp_end]
)
     VALUES
           (
           '2018-01-01 00:00:00'
           ,'2019-12-31 23:59:59' 
)


go


if (OBJECT_ID('[Dflt].SpadeRF_00_Gen_00_ICD10Code') is not null)
	drop table [Dflt].SpadeRF_00_Gen_00_ICD10Code

CREATE TABLE [Dflt].[SpadeRF_00_Gen_00_ICD10Code](
	[CategoryName] [varchar](50) NULL,
	[ICD10Code] [varchar](50) NULL,
	[ICD10Description] [varchar](255) NULL
)
go

insert into [Dflt].[SpadeRF_00_Gen_00_ICD10Code](
	[CategoryName],
	[ICD10Code],
	[ICD10Description] 
)
values
 ('Stroke','G43.601',null) 
,('Stroke','G43.609',null) 
,('Stroke','G43.611',null) 
,('Stroke','G43.619',null) 
,('Stroke','G45.9',null) 
,('Stroke','I60.00',null) 
,('Stroke','I60.01',null) 
,('Stroke','I60.02',null) 
,('Stroke','I60.10',null) 
,('Stroke','I60.11',null) 
,('Stroke','I60.12',null) 
,('Stroke','I60.2',null) 
,('Stroke','I60.20',null) 
,('Stroke','I60.21',null) 
,('Stroke','I60.22',null) 
,('Stroke','I60.30',null) 
,('Stroke','I60.31',null) 
,('Stroke','I60.32',null) 
,('Stroke','I60.4',null) 
,('Stroke','I60.50',null) 
,('Stroke','I60.51',null) 
,('Stroke','I60.52',null) 
,('Stroke','I60.6',null) 
,('Stroke','I60.7',null) 
,('Stroke','I60.8',null) 
,('Stroke','I60.9',null) 
,('Stroke','I61.0',null) 
,('Stroke','I61.1',null) 
,('Stroke','I61.2',null) 
,('Stroke','I61.3',null) 
,('Stroke','I61.4',null) 
,('Stroke','I61.5',null) 
,('Stroke','I61.6',null) 
,('Stroke','I61.8',null) 
,('Stroke','I61.9',null) 
,('Stroke','I62.00',null) 
,('Stroke','I62.01',null) 
,('Stroke','I62.02',null) 
,('Stroke','I62.03',null) 
,('Stroke','I62.1',null) 
,('Stroke','I62.9',null) 
,('Stroke','I63.00',null) 
,('Stroke','I63.011',null) 
,('Stroke','I63.012',null) 
,('Stroke','I63.013',null) 
,('Stroke','I63.019',null) 
,('Stroke','I63.02',null) 
,('Stroke','I63.031',null) 
,('Stroke','I63.032',null) 
,('Stroke','I63.033',null) 
,('Stroke','I63.039',null) 
,('Stroke','I63.09',null) 
,('Stroke','I63.10',null) 
,('Stroke','I63.111',null) 
,('Stroke','I63.112',null) 
,('Stroke','I63.113',null) 
,('Stroke','I63.119',null) 
,('Stroke','I63.12',null) 
,('Stroke','I63.131',null) 
,('Stroke','I63.132',null) 
,('Stroke','I63.133',null) 
,('Stroke','I63.139',null) 
,('Stroke','I63.19',null) 
,('Stroke','I63.20',null) 
,('Stroke','I63.211',null) 
,('Stroke','I63.212',null) 
,('Stroke','I63.213',null) 
,('Stroke','I63.219',null) 
,('Stroke','I63.22',null) 
,('Stroke','I63.231',null) 
,('Stroke','I63.232',null) 
,('Stroke','I63.233',null) 
,('Stroke','I63.239',null) 
,('Stroke','I63.29',null) 
,('Stroke','I63.30',null) 
,('Stroke','I63.311',null) 
,('Stroke','I63.312',null) 
,('Stroke','I63.313',null) 
,('Stroke','I63.319',null) 
,('Stroke','I63.321',null) 
,('Stroke','I63.322',null) 
,('Stroke','I63.323',null) 
,('Stroke','I63.329',null) 
,('Stroke','I63.331',null) 
,('Stroke','I63.332',null) 
,('Stroke','I63.333',null) 
,('Stroke','I63.339',null) 
,('Stroke','I63.341',null) 
,('Stroke','I63.342',null) 
,('Stroke','I63.343',null) 
,('Stroke','I63.349',null) 
,('Stroke','I63.39',null) 
,('Stroke','I63.40',null) 
,('Stroke','I63.411',null) 
,('Stroke','I63.412',null) 
,('Stroke','I63.413',null) 
,('Stroke','I63.419',null) 
,('Stroke','I63.421',null) 
,('Stroke','I63.422',null) 
,('Stroke','I63.423',null) 
,('Stroke','I63.429',null) 
,('Stroke','I63.431',null) 
,('Stroke','I63.432',null) 
,('Stroke','I63.433',null) 
,('Stroke','I63.439',null) 
,('Stroke','I63.441',null) 
,('Stroke','I63.442',null) 
,('Stroke','I63.443',null) 
,('Stroke','I63.449',null) 
,('Stroke','I63.49',null) 
,('Stroke','I63.50',null) 
,('Stroke','I63.511',null) 
,('Stroke','I63.512',null) 
,('Stroke','I63.513',null) 
,('Stroke','I63.519',null) 
,('Stroke','I63.521',null) 
,('Stroke','I63.522',null) 
,('Stroke','I63.523',null) 
,('Stroke','I63.529',null) 
,('Stroke','I63.531',null) 
,('Stroke','I63.532',null) 
,('Stroke','I63.533',null) 
,('Stroke','I63.539',null) 
,('Stroke','I63.541',null) 
,('Stroke','I63.542',null) 
,('Stroke','I63.543',null) 
,('Stroke','I63.549',null) 
,('Stroke','I63.59',null) 
,('Stroke','I63.6',null) 
,('Stroke','I63.8',null) 
,('Stroke','I63.81',null) 
,('Stroke','I63.89',null) 
,('Stroke','I63.9',null) 

insert into [Dflt].[SpadeRF_00_Gen_00_ICD10Code](
	[CategoryName],
	[ICD10Code],
	[ICD10Description] 
)
values
 ('Dizziness','H81.01',null) 
,('Dizziness','H81.02',null) 
,('Dizziness','H81.03',null) 
,('Dizziness','H81.09',null) 
,('Dizziness','H81.10',null) 
,('Dizziness','H81.11',null) 
,('Dizziness','H81.12',null) 
,('Dizziness','H81.13',null) 
,('Dizziness','H81.20',null) 
,('Dizziness','H81.21',null) 
,('Dizziness','H81.22',null) 
,('Dizziness','H81.23',null) 
,('Dizziness','H81.311',null) 
,('Dizziness','H81.312',null) 
,('Dizziness','H81.313',null) 
,('Dizziness','H81.319',null) 
,('Dizziness','H81.391',null) 
,('Dizziness','H81.392',null) 
,('Dizziness','H81.393',null) 
,('Dizziness','H81.399',null) 
,('Dizziness','H81.4',null) 
,('Dizziness','H81.41',null) 
,('Dizziness','H81.42',null) 
,('Dizziness','H81.43',null) 
,('Dizziness','H81.49',null) 
,('Dizziness','H81.8X1',null) 
,('Dizziness','H81.8X2',null) 
,('Dizziness','H81.8X3',null) 
,('Dizziness','H81.8X9',null) 
,('Dizziness','H81.90',null) 
,('Dizziness','H81.91',null) 
,('Dizziness','H81.92',null) 
,('Dizziness','H81.93',null) 
,('Dizziness','H83.01',null) 
,('Dizziness','H83.02',null) 
,('Dizziness','H83.03',null) 
,('Dizziness','H83.09',null) 
,('Dizziness','H83.11',null) 
,('Dizziness','H83.12',null) 
,('Dizziness','H83.13',null) 
,('Dizziness','H83.19',null) 
,('Dizziness','H83.2X1',null) 
,('Dizziness','H83.2X2',null) 
,('Dizziness','H83.2X3',null) 
,('Dizziness','H83.2X9',null) 
,('Dizziness','H83.3X1',null) 
,('Dizziness','H83.3X2',null) 
,('Dizziness','H83.3X3',null) 
,('Dizziness','H83.3X9',null) 
,('Dizziness','H83.8X1',null) 
,('Dizziness','H83.8X2',null) 
,('Dizziness','H83.8X3',null) 
,('Dizziness','H83.8X9',null) 
,('Dizziness','H83.90',null) 
,('Dizziness','H83.91',null) 
,('Dizziness','H83.92',null) 
,('Dizziness','H83.93',null) 
,('Dizziness','R42.',null)
-------------
,('Headache','G43.001',null) 
,('Headache','G43.009',null) 
,('Headache','G43.011',null) 
,('Headache','G43.019',null) 
,('Headache','G43.101',null) 
,('Headache','G43.109',null) 
,('Headache','G43.111',null) 
,('Headache','G43.119',null) 
,('Headache','G43.401',null) 
,('Headache','G43.409',null) 
,('Headache','G43.411',null) 
,('Headache','G43.419',null) 
,('Headache','G43.501',null) 
,('Headache','G43.509',null) 
,('Headache','G43.511',null) 
,('Headache','G43.519',null) 
,('Headache','G43.601',null) 
,('Headache','G43.609',null) 
,('Headache','G43.611',null) 
,('Headache','G43.619',null) 
,('Headache','G43.701',null) 
,('Headache','G43.709',null) 
,('Headache','G43.711',null) 
,('Headache','G43.719',null) 
,('Headache','G43.801',null) 
,('Headache','G43.809',null) 
,('Headache','G43.811',null) 
,('Headache','G43.819',null) 
,('Headache','G43.821',null) 
,('Headache','G43.829',null) 
,('Headache','G43.831',null) 
,('Headache','G43.839',null) 
,('Headache','G43.901',null) 
,('Headache','G43.909',null) 
,('Headache','G43.911',null) 
,('Headache','G43.919',null) 
,('Headache','G43.A0',null) 
,('Headache','G43.A1',null) 
,('Headache','G43.B0',null) 
,('Headache','G43.B1',null) 
,('Headache','G43.C0',null) 
,('Headache','G43.C1',null) 
,('Headache','G43.D0',null) 
,('Headache','G43.D1',null) 
,('Headache','G44.001',null) 
,('Headache','G44.009',null) 
,('Headache','G44.011',null) 
,('Headache','G44.019',null) 
,('Headache','G44.021',null) 
,('Headache','G44.029',null) 
,('Headache','G44.031',null) 
,('Headache','G44.039',null) 
,('Headache','G44.041',null) 
,('Headache','G44.049',null) 
,('Headache','G44.051',null) 
,('Headache','G44.059',null) 
,('Headache','G44.091',null) 
,('Headache','G44.099',null) 
,('Headache','G44.1',null) 
,('Headache','G44.201',null) 
,('Headache','G44.209',null) 
,('Headache','G44.211',null) 
,('Headache','G44.219',null) 
,('Headache','G44.221',null) 
,('Headache','G44.229',null) 
,('Headache','G44.301',null) 
,('Headache','G44.309',null) 
,('Headache','G44.311',null) 
,('Headache','G44.319',null) 
,('Headache','G44.321',null) 
,('Headache','G44.329',null) 
,('Headache','G44.40',null) 
,('Headache','G44.41',null) 
,('Headache','G44.51',null) 
,('Headache','G44.52',null) 
,('Headache','G44.53',null) 
,('Headache','G44.59',null) 
,('Headache','G44.81',null) 
,('Headache','G44.82',null) 
,('Headache','G44.83',null) 
,('Headache','G44.84',null) 
,('Headache','G44.85',null) 
,('Headache','G44.89',null) 
,('Headache','R51.',null)

if (OBJECT_ID('[Dflt].SpadeRF_00_Gen_00_ICD9Code') is not null)
	drop table [Dflt].SpadeRF_00_Gen_00_ICD9Code

CREATE TABLE [Dflt].[SpadeRF_00_Gen_00_ICD9Code](
	[CategoryName] [varchar](50) NULL,
	[ICD9Code] [varchar](50) NULL,
	[ICD9Description] [varchar](255) NULL
)
go

insert into [Dflt].[SpadeRF_00_Gen_00_ICD9Code](
	[CategoryName],
	[ICD9Code],
	[ICD9Description] 
)
values
('Stroke','346.60',null) 
,('Stroke','346.61',null) 
,('Stroke','346.62',null) 
,('Stroke','346.63',null) 
,('Stroke','430.',null)
,('Stroke','431.',null)
,('Stroke','432.0',null)
,('Stroke','432.1',null)
,('Stroke','432.9',null)
,('Stroke','433.01',null) 
,('Stroke','433.11',null) 
,('Stroke','433.21',null) 
,('Stroke','433.31',null) 
,('Stroke','433.81',null) 
,('Stroke','433.91',null) 
,('Stroke','434.00',null) 
,('Stroke','434.00',null) 
,('Stroke','434.10',null) 
,('Stroke','434.90',null) 
,('Stroke','434.90',null)
,('Stroke','434.01',null) 
,('Stroke','434.10',null) 
,('Stroke','434.11',null) 
,('Stroke','434.91',null) 
,('Stroke','436.',null)
,('Stroke','430.',null)
,('Stroke','431.',null)
,('Stroke','432.0',null)
,('Stroke','432.1',null)
,('Stroke','432.9',null)
,('Stroke','433.01',null) 
,('Stroke','433.11',null) 
,('Stroke','433.21',null) 
,('Stroke','433.31',null) 
,('Stroke','433.81',null) 
,('Stroke','433.91',null) 
,('Stroke','434.0',null)
,('Stroke','434.00',null) 
,('Stroke','434.01',null) 
,('Stroke','434.1',null)
,('Stroke','434.10',null) 
,('Stroke','434.11',null) 
,('Stroke','434.9',null)
,('Stroke','434.90',null) 
,('Stroke','434.91',null) 
,('Stroke','346.60',null) 
,('Stroke','346.61',null) 
,('Stroke','346.62',null) 
,('Stroke','346.63',null) 
,('Stroke','436.',null)
,('Stroke','433.0',null)
,('Stroke','433.1',null)
,('Stroke','433.10',null) 
,('Stroke','433.2',null)
,('Stroke','433.20',null)
,('Stroke','433.3',null)
,('Stroke','433.30',null) 
,('Stroke','433.8',null)
,('Stroke','433.80',null) 
,('Stroke','433.9',null)
,('Stroke','433.90',null) 
,('Stroke','437.0',null)
,('Stroke','437.1',null)
,('Stroke','437.3',null)
,('Stroke','437.4',null)
,('Stroke','437.5',null)
,('Stroke','437.6',null)
,('Stroke','437.7',null)
,('Stroke','437.8',null)
,('Stroke','437.9',null) 
,('Stroke','435.0',null)
,('Stroke','435.1',null)
,('Stroke','435.2',null)
,('Stroke','435.3',null)
,('Stroke','435.8',null)
,('Stroke','435.9',null)

insert into [Dflt].[SpadeRF_00_Gen_00_ICD9Code](
	[CategoryName],
	[ICD9Code],
	[ICD9Description] 
)
values 
('Dizziness','386.00',null) 
,('Dizziness','386.01',null) 
,('Dizziness','386.02',null) 
,('Dizziness','386.03',null) 
,('Dizziness','386.04',null) 
,('Dizziness','386.10',null) 
,('Dizziness','386.11',null) 
,('Dizziness','386.12',null) 
,('Dizziness','386.19',null) 
,('Dizziness','386.2',null)
,('Dizziness','386.30',null) 
,('Dizziness','386.31',null) 
,('Dizziness','386.32',null) 
,('Dizziness','386.33',null) 
,('Dizziness','386.34',null) 
,('Dizziness','386.35',null) 
,('Dizziness','386.40',null) 
,('Dizziness','386.41',null) 
,('Dizziness','386.42',null) 
,('Dizziness','386.43',null) 
,('Dizziness','386.48',null) 
,('Dizziness','386.50',null) 
,('Dizziness','386.51',null) 
,('Dizziness','386.52',null) 
,('Dizziness','386.53',null) 
,('Dizziness','386.54',null) 
,('Dizziness','386.55',null) 
,('Dizziness','386.56',null) 
,('Dizziness','386.58',null) 
,('Dizziness','386.8',null)
,('Dizziness','386.9',null)
,('Dizziness','780.4',null)
,('Dizziness','380.00',null) 
,('Dizziness','380.01',null) 
,('Dizziness','380.02',null) 
,('Dizziness','380.03',null) 
,('Dizziness','380.10',null) 
,('Dizziness','380.11',null) 
,('Dizziness','380.12',null) 
,('Dizziness','380.13',null) 
,('Dizziness','380.14',null) 
,('Dizziness','380.15',null) 
,('Dizziness','380.16',null) 
,('Dizziness','380.21',null) 
,('Dizziness','380.22',null) 
,('Dizziness','380.23',null) 
,('Dizziness','380.30',null) 
,('Dizziness','380.31',null) 
,('Dizziness','380.32',null) 
,('Dizziness','380.39',null) 
,('Dizziness','380.4',null)
,('Dizziness','380.50',null) 
,('Dizziness','380.51',null) 
,('Dizziness','380.52',null) 
,('Dizziness','380.53',null) 
,('Dizziness','380.81',null) 
,('Dizziness','380.89',null) 
,('Dizziness','380.9',null)
,('Dizziness','400.',null)
,('Dizziness','384.01',null) 
,('Dizziness','384.09',null) 
,('Dizziness','384.1',null)
,('Dizziness','385.30',null) 
,('Dizziness','385.31',null) 
,('Dizziness','385.32',null) 
,('Dizziness','385.33',null) 
,('Dizziness','385.35',null) 
,('Dizziness','385.82',null) 
,('Dizziness','385.83',null) 
,('Dizziness','385.89',null) 
,('Dizziness','385.9',null)
,('Dizziness','388.00',null) 
,('Dizziness','388.01',null) 
,('Dizziness','388.02',null) 
,('Dizziness','388.10',null) 
,('Dizziness','388.11',null) 
,('Dizziness','388.12',null) 
,('Dizziness','388.2',null)
,('Dizziness','388.30',null) 
,('Dizziness','388.31',null) 
,('Dizziness','388.32',null) 
,('Dizziness','388.40',null) 
,('Dizziness','388.41',null) 
,('Dizziness','388.42',null) 
,('Dizziness','388.43',null) 
,('Dizziness','388.44',null) 
,('Dizziness','388.45',null) 
,('Dizziness','388.5',null)
,('Dizziness','388.60',null) 
,('Dizziness','388.61',null) 
,('Dizziness','388.69',null) 
,('Dizziness','388.70',null) 
,('Dizziness','388.71',null) 
,('Dizziness','388.72',null) 
,('Dizziness','388.8',null)
,('Dizziness','388.9',null)
,('Dizziness','389.00',null) 
,('Dizziness','389.01',null) 
,('Dizziness','389.02',null) 
,('Dizziness','389.03',null) 
,('Dizziness','389.04',null) 
,('Dizziness','389.08',null)
,('Dizziness','389.05',null) 
,('Dizziness','389.06',null) 
,('Dizziness','389.10',null) 
,('Dizziness','389.16',null)
,('Dizziness','389.11',null) 
,('Dizziness','389.12',null) 
,('Dizziness','389.18',null)
,('Dizziness','389.13',null) 
,('Dizziness','389.17',null)
,('Dizziness','389.14',null) 
,('Dizziness','389.15',null)  
,('Dizziness','389.2',null)
,('Dizziness','389.20',null) 
,('Dizziness','389.21',null)
,('Dizziness','389.22',null) 
,('Dizziness','389.7',null)
,('Dizziness','389.8',null)
,('Dizziness','389.9',null)
,('Dizziness','V41.2',null)
,('Dizziness','V41.3',null)
,('Dizziness','V49.85',null) 
,('Dizziness','V53.2',null)
,('Dizziness','V72.1',null)
,('Dizziness','V72.11',null) 
,('Dizziness','V72.12',null) 
,('Dizziness','V72.19',null) 
-------------
,('Headache','339.00',null) 
,('Headache','339.01',null) 
,('Headache','339.02',null) 
,('Headache','339.03',null) 
,('Headache','339.04',null) 
,('Headache','346.0',null)
,('Headache','346.00',null) 
,('Headache','346.01',null) 
,('Headache','346.02',null) 
,('Headache','346.03',null) 
,('Headache','346.1',null)
,('Headache','346.10',null) 
,('Headache','346.11',null) 
,('Headache','346.12',null) 
,('Headache','346.13',null) 
,('Headache','346.2',null)
,('Headache','346.20',null) 
,('Headache','346.21',null) 
,('Headache','346.22',null) 
,('Headache','346.23',null) 
,('Headache','346.30',null) 
,('Headache','346.31',null) 
,('Headache','346.32',null) 
,('Headache','346.33',null) 
,('Headache','346.40',null) 
,('Headache','346.41',null) 
,('Headache','346.42',null) 
,('Headache','346.43',null) 
,('Headache','346.50',null) 
,('Headache','346.51',null) 
,('Headache','346.52',null)
,('Headache','346.53',null) 
,('Headache','346.70',null) 
,('Headache','346.71',null) 
,('Headache','346.72',null) 
,('Headache','346.73',null) 
,('Headache','346.8',null)
,('Headache','346.80',null) 
,('Headache','346.81',null) 
,('Headache','346.82',null) 
,('Headache','346.83',null) 
,('Headache','346.9',null)
,('Headache','346.90',null) 
,('Headache','346.91',null) 
,('Headache','346.92',null) 
,('Headache','346.93',null)
,('Headache','339.05',null) 
,('Headache','339.09',null) 
,('Headache','339.10',null) 
,('Headache','339.11',null) 
,('Headache','339.12',null) 
,('Headache','339.20',null) 
,('Headache','339.21',null) 
,('Headache','339.22',null) 
,('Headache','339.3',null)
,('Headache','339.41',null) 
,('Headache','339.42',null) 
,('Headache','339.43',null) 
,('Headache','339.44',null) 
,('Headache','339.81',null) 
,('Headache','339.82',null) 
,('Headache','339.83',null) 
,('Headache','339.84',null) 
,('Headache','339.85',null) 
,('Headache','339.89',null) 
,('Headache','784.0',null)
-- newly added:
,('Dizziness','078.81',null) 
,('Headache','784.0',null) 
,('Headache','307.81',null) 
,('Dizziness','438.85',null)


if (OBJECT_ID('[Dflt].[SpadeRF_010_01_InpatStrokeWithPrior30DayER]') is not null)	
	drop table [Dflt].SpadeRF_010_01_InpatStrokeWithPrior30DayER
go

select 
 	a.inpatientsid,
	a.InpatientDischargeDiagnosisSID,
	a.patientsid,
	a.Sta3n,
	a.AdmitDateTime,
	InpatIcd.ICD10Code as InpatICD10Code,
	InpatIcdDes.ICD10Description as InpatICD10Description,
	isnull(InpatIcd9.ICD9Code,'') as InpatICD9Code,
	isnull(InpatIcd9Des.ICD9Description,'') as InpatICD9Description,
	PatientArrivalDateTime,
	PatientDepartureDateTime,
    PatientVisitReason,
	lg.visitsid as ERVisitSID,
	DispositionEDISTrackingCodeSID,
	TRACKINGCODENAME,
	EDISTrackingCodeSID,	
    SECTIONDISPLAYNAME, 
    DISPLAYnAMEaBBREVIATION,
	EDISLogSID
into  [dflt].[SpadeRF_010_01_InpatStrokeWithPrior30DayER] 
  FROM [Src].Inpat_InpatDischargeDiagnosis a
  left join CDWWork.dim.ICD10 as InpatIcd
  on a.ICD10SID=InpatIcd.ICD10SID 
  left join CDWWork.dim.ICD10DescriptionVersion as InpatIcdDes
  on InpatIcd.ICD10SID=InpatIcdDes.ICD10SID
  left join CDWWork.dim.ICD9 as InpatIcd9
  on a.ICD9SID=InpatIcd9.ICD9SID 
  left join CDWWork.dim.ICD9DescriptionVersion as InpatIcd9Des
  on InpatIcd9.ICD9SID=InpatIcd9Des.ICD9SID
  left join [Src].[EDIS_EDISLog] as lg
  on a.sta3n=lg.sta3n and a.patientsid=lg.patientsid
  left join cdwwork.Dim.EDISTrackingCode as tc
  on lg.DispositionEDISTrackingCodeSID=tc.EDISTrackingCodeSID
  where AdmitDateTime between (select sp_start from [Dflt].SpadeRF_00_1_inputP)
						  and (select sp_end from [Dflt].SpadeRF_00_1_inputP)   
  and ( isnull(InpatIcd.ICD10Code,'') in              
			(select [ICD10Code] from [Dflt].[SpadeRF_00_Gen_00_ICD10Code] 
		     where [CategoryName]='Stroke')
	  or isnull(InpatIcd9.ICD9Code,'') in         													 
			(select [ICD9Code] from [Dflt].[SpadeRF_00_Gen_00_ICD9Code] 
		     where [CategoryName]='Stroke')     
	  )  
and lg.PatientDepartureDateTime between dateadd(day,-30, AdmitDateTime) and AdmitDateTime
go



if (OBJECT_ID('[Dflt].[SpadeRF_020_02_InpatStrokeWithPrior30DayER_Outpat]') is not null)	
	drop table [Dflt].SpadeRF_020_02_InpatStrokeWithPrior30DayER_Outpat
go

select 
	a.*,
	vDiag.VisitSID as outpatVDiagVisitSID,
	vis.VisitSID as outpatVisitSID
	,vis.VisitDateTime
    ,EDICD.ICD10Code as ERICD10Code,EDICDDes.ICD10Description as ERICD10Description
    ,EDICD9.ICD9Code as ERICD9Code,EDICD9Des.ICD9Description as ERICD9Description
	,vDiag.vDiagnosisSID
	,vDiag.ProblemListSID
into [dflt].[SpadeRF_020_02_InpatStrokeWithPrior30DayER_Outpat]
from [dflt].[SpadeRF_010_01_InpatStrokeWithPrior30DayER]  as a
  left join [Src].outpat_vDiagnosis as vDiag 
  on (vDiag.VisitSID=ERVisitSID and vDiag.VisitSID<>-1 )
    left join [Src].outpat_visit as vis  
  on (vis.VisitSID=ERVisitSID and vis.VisitSID<>-1 )  
  left join CDWWork.dim.ICD10 as EDICD
  on vDiag.ICD10SID=EDICD.ICD10SID
  left join CDWWork.dim.ICD10DescriptionVersion as EDICDDes
  on EDICD.ICD10SID=EDICDDes.ICD10SID 
  left join CDWWork.dim.ICD9 as EDICD9
  on vDiag.ICD9SID=EDICD9.ICD9SID
  left join CDWWork.dim.ICD9DescriptionVersion as EDICD9Des
  on EDICD9.ICD9SID=EDICD9Des.ICD9SID 
where not(PatientDepartureDateTime is  null  
  or [ERVisitSID] =-1 or [ERVisitSID] is null  
  or vis.VisitSID is null or vis.VisitSID =-1                   
  or vDiag.VisitSID is null               
  )
 go
     

if (OBJECT_ID('[Dflt].[SpadeRF_030_01_InpatStrokeWithPrior30DayER_Outpat_OnlyDizzHeadache]') is not null)	
	drop table [Dflt].SpadeRF_030_01_InpatStrokeWithPrior30DayER_Outpat_OnlyDizzHeadache
go

  select * 
  into [Dflt].[SpadeRF_030_01_InpatStrokeWithPrior30DayER_Outpat_OnlyDizzHeadache]  
  from [dflt].[SpadeRF_020_02_InpatStrokeWithPrior30DayER_Outpat] as a
  where (
  isnull(ERICD10code,'') in (select [ICD10Code] from [Dflt].[SpadeRF_00_Gen_00_ICD10Code]
					where [CategoryName] in ('Dizziness','Headache'))
or isnull(ERICD9code,'') in (select [ICD9Code] from [Dflt].[SpadeRF_00_Gen_00_ICD9Code]
					where [CategoryName] in ('Dizziness','Headache'))
			)
 and not exists(
 --from vDiagnosis
		select * from (
			    select x.inpatientsid,sta3n,patientsid,outpatVisitSID
					   ,x.ERICD10Code as ERStrokeICD10Code,x.ERICD10Description as ERStrokeICD10Description 				
				from  [dflt].[SpadeRF_020_02_InpatStrokeWithPrior30DayER_Outpat] as x
				where (isnull(ERICD10code,'') in (select [ICD10Code] from [Dflt].[SpadeRF_00_Gen_00_ICD10Code]
										where [CategoryName] in ('Stroke'))
					   or isnull(ERICD9code,'') in (select [ICD9Code] from [Dflt].[SpadeRF_00_Gen_00_ICD9Code]												    
										where [CategoryName] in ('Stroke'))
										)
		) as b
    where a.inpatientsid=b.inpatientsid and a.outpatVisitSID=b.outpatVisitSID  

 ---- from problemlist
 --        select * from (
	--			select x.inpatientsid,x.sta3n,x.patientsid,outpatVisitSID--,x.vDiagnosisSID
	--					,c.ICD10Code as ERStrokeICD10CodeFromProblemList,f.ICD10Description as ERStrokeICD10DescriptionFromProblemList 
	--			from  [dflt].[SpadeRF_020_02_InpatStrokeWithPrior30DayER_Outpat] as x
	--			inner join src.Outpat_ProblemList as y
	--			on x.ProblemListSID=y.ProblemListSID and y.cohortname='Cohort20180712'
	--		  inner join cdwwork.dim.ICD10 as c
	--		  on y.ICD10SID=c.ICD10SID 
	--		  inner join CDWWork.dim.ICD10DescriptionVersion as f
	--		  on c.ICD10SID=f.ICD10SID
	--		  where c.ICD10Code in (select [ICD10Code] from [Dflt].[SpadeRF_00_Gen_00_ICD10Code]
	--									where [CategoryName] in ('Stroke') )
	--	) as b
	--	where a.inpatientsid=b.inpatientsid and a.outpatVisitSID=b.outpatVisitSID
 )

if (OBJECT_ID('[Dflt].[SpadeRF_030_02_InpatStrokeWithPrior30DayE_Outpat_OnlyDizzHeadache_sentHome]') is not null)	
	drop table [Dflt].SpadeRF_030_02_InpatStrokeWithPrior30DayE_Outpat_OnlyDizzHeadache_sentHome
go

SELECT [inpatientsid]
	,b.PatientSSN
      ,a.[patientsid]
      ,a.[Sta3n]
      ,[AdmitDateTime]
	  ,datediff(hour,[PatientDepartureDateTime],admitdatetime) as withinHour
      ,[InpatICD10Code]
      ,[InpatICD10Description]
      ,[InpatICD9Code]
      ,[InpatICD9Description]
      ,[PatientArrivalDateTime]
      ,[PatientDepartureDateTime]
      ,[PatientVisitReason]
      ,[SECTIONDISPLAYNAME]
      ,[outpatVisitSID]
      ,[ERICD10Code]
      ,[ERICD10Description]
      ,[ERICD9Code]
      ,[ERICD9Description]
	  ,vDiagnosisSID
	  ,ProblemListSID
into [dflt].[SpadeRF_030_02_InpatStrokeWithPrior30DayE_Outpat_OnlyDizzHeadache_sentHome]
from [dflt].[SpadeRF_030_01_InpatStrokeWithPrior30DayER_Outpat_OnlyDizzHeadache] as a
left join Src.SPatient_SPatient as b
on a.patientsid=b.patientsid and a.sta3n=b.sta3n
  where 
  (SECTIONDISPLAYNAME like '%Home%' or 
DISPLAYnAMEaBBREVIATION ='H'
)
  order by a.sta3n,a.patientsid,admitdatetime,[PatientDepartureDateTime]


  select *
  --select distinct patientssn  --282

   from [dflt].[SpadeRF_030_02_InpatStrokeWithPrior30DayE_Outpat_OnlyDizzHeadache_sentHome] 
   where year(admitDateTime) in (2018,2019)
 