f = open('testForGRU.txt', 'w')
f.write("")
f.close()
f = open('data/testTrainData.txt', 'a')

for line in open ("data/trainTestData.txt"):
	s = line.split()
	for i in range(0, len(s)):
		r = str(s[i] + '\t' + " ".join(str(x) for x in s[:i]))
		f.write(r + '\n')
	#r = str("</s>" + '\t' + " ".join(str(x) for x in s))
	#f.write(r + '\n')
f.close()