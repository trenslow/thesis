import yaml

# at some point, needs to be generalized to all files
lang_pairs = [('bul', 'rus'), ('ces', 'pol'), ('rus', 'bul'), ('pol', 'ces')]
lang_map = {'bul': 'BG', 'ces': 'CS', 'pol': 'PL', 'rus': 'RU'}
raw_data_dir = 'data/raw/'
processed_data_dir = 'data/processed/'

for pair in lang_pairs:
	lang1, lang2 = pair[0], pair[1]
	andrea_lang1, andrea_lang2 = lang_map[lang1], lang_map[lang2]
	in_file = raw_data_dir + andrea_lang1 + '-' + andrea_lang2 + '_reconstructed.yaml'
	out_file = processed_data_dir + 'clusters_reconstructed.' + lang1 + '.' + lang2
	with open(in_file) as f, open(out_file, 'w+') as out:
		data = yaml.load(f)
		for head, alignments in data.items():
			for alignment in alignments:
				left_word, right_word = '', ''
				for left, right in alignment:
					left_word += ''.join(left)
					right_word += ''.join(right)
				# direction of recon not consistent across recon files
				if lang1 == 'ces' or lang1 == 'rus':  # ces and rus have left to right recon
					out.write(left_word + ' ' + right_word + '\n')
				else:  # bul and pol have right to left
					out.write(right_word + ' ' + left_word + '\n')

			