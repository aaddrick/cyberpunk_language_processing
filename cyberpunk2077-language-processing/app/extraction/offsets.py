def find_offsets(sig, archive_path):
	prev = b''
	decimals = []
	with open(archive_path, 'rb') as f:
		if f.read(4) != b'\x52\x44\x41\x52':
			raise Exception('Not a valid archive file.')
		f.seek(0)
		while True:
			concat_pos = 0
			buf = f.read(max(2048 ** 2, len(sig)))
			if not buf:
				break
			concat = prev + buf
			while True:
				concat_pos = concat.find(sig, concat_pos)
				if concat_pos == -1:
					break
				pos = f.tell() + concat_pos - len(concat)
				if sig == b'\x52\x49\x46\x46':
					cur_pos = f.tell()
					f.seek(pos + 16)
					_byte = f.read(1)
					f.seek(cur_pos)
					if _byte != b'\x42':
						concat_pos += len(sig)
						continue
				decimals.append(pos)
				concat_pos += len(sig)
			prev = buf[-len(sig) + 1:]
	return decimals