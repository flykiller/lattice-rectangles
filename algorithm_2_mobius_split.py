from math import gcd

def sum_h1w1(h, w, s, r):
	m = 1 + min((h - 1) // s, (w - 1) // r)
	sk, sk2 = m * (m - 1) // 2, (m - 1) * m * (2 * m - 1) // 6
	return m * h * w - (h * r + w * s) * sk + r * s * sk2

def build_divs_mu(n):
	out = [[] for _ in range(n + 1)]; out[1] = [(1, 1)]
	for x in range(2, n + 1):
		y, ps, d = x, [], 2
		while d * d <= y:
			if y % d == 0: ps.append(d)
			while y % d == 0: y //= d
			d += 1
		if y > 1: ps.append(y)
		cur = [(1, 1)]
		for p in ps: cur += [(d * p, -m) for d, m in cur]
		out[x] = cur
	return out

def coprime_interval_sums(divs, L, U):
	cnt = s1 = s2 = 0
	for d, mu in divs:
		lo = (L + d - 1)//d; hi = U//d
		cnt += mu * (hi - lo + 1)
		s1 += mu * d * (hi * (hi + 1)//2 - (lo - 1) * lo // 2)
		s2 += mu * d * d * (hi * (hi + 1) * (2 * hi + 1) - (lo - 1) * lo * (2 * lo - 1)) // 6
	return cnt, s1, s2

def small_part(n, B, residues):
	total = 0
	for s in range(1, B):
		start = 2 if s == 1 else s
		for a in residues[s]:
			for r in range(a + ((start - a + s - 1) // s) * s, n - s, s):
				h0 = n - r - s
				for k in range(1 + min((h0 - 1)//r, (h0 - 1)//s)):
					total += sum_h1w1(h0 - k*r, h0 - k*s, s, r)
	return total

def large_part(n, B, divs_mu):
	total = 0
	for t in range((n - 1) // B - 1 + 1):
		a = t + 1
		for u in range(min(t, (n - 1) // B - t - 2) + 1):
			mult = 1 if t == u else 2
			b = u + 1
			for r in range(max(B, 2), (n - 1 - b * B) // a + 1):
				cnt, s1, s2 = coprime_interval_sums(divs_mu[r], B, min(r, (n - 1 - a * r) // b))
				n1, n2 = n - a * r, n - b * r
				total += mult * (n1 * n2 * cnt - (a * n1 + b * n2) * s1 + a * b * s2)
	return total

def rect_fastest(n):
	B = int(n**0.5)
	divs_mu = build_divs_mu(n)
	residues = [[]] + [[a for a in range(1, s + 1) if gcd(a, s) == 1] for s in range(1, B)]
	base = (n - 1) ** 2 * n * (2 * n - 1) // 6
	return base + 2 * (small_part(n, B, residues) + large_part(n, B, divs_mu))