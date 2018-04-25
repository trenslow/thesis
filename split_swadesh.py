import csv

data_dir = 'data/'
swadesh_list_file = data_dir + 'Slavic_Swadesh.csv'
swadesh_bul_file = data_dir + 'swadesh.bul'
swadesh_ces_file = data_dir	+ 'swadesh.ces'
swadesh_pol_file = data_dir + 'swadesh.pol'
swadesh_rus_file = data_dir + 'swadesh.rus'

with open(swadesh_list_file) as csvfile, open(swadesh_bul_file, 'w+') as bul, open(swadesh_ces_file, 'w+') as ces, open(swadesh_pol_file, 'w+') as pol, open(swadesh_rus_file, 'w+') as rus:
	swadesh_reader = csv.DictReader(csvfile)
	for row in swadesh_reader:
		bul.write(row['BG'] + '\n')
		ces.write(row['CS'] + '\n')
		pol.write(row['PL'] + '\n')
		rus.write(row['RU'] + '\n')