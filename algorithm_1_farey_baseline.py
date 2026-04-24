def farey_sequence(n):
	a, b, c, d = 0, 1, 1, n
	yield (a, b)
	while c <= n:
		k = (n + b) // d
		a, b, c, d = c, d, k * c - a, k * d - b
		yield (a, b)

def sum_h1w1(h, w, s, r):
	m1 = (h - 1) // s
	m2 = (w - 1) // r
	m = 1 + min(m1, m2)
	sk = m * (m - 1) // 2
	sk2 = (m - 1) * m * (2 * m - 1) // 6
	return m * h * w - (h * r + w * s) * sk + (r * s) * sk2

def F_fast(n):
	base = (n - 1) ** 2 * n * (2 * n - 1) // 6
	count = 0
	for s, r in farey_sequence(n - 1):
		if r == 1: continue
		h = w = n - r - s
		while h > 0 and w > 0:
			count += sum_h1w1(h, w, s, r)
			h -= r
			w -= s
	return base + 2 * count