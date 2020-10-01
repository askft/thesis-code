sentence = "Four groups were used , i ) the PAN group ( 14 ) , ii ) PAN / temocapril ( 13 ) , iii ) temocapril ( 14 ) and iv ) untreated controls ( 15 ) ."
labels = ["O", "O", "O", "O", "O", "O", "O", "O", "B", "O", "O", "O", "O", "O", "O", "O", "B", "O", "B", "O", "O", "O", "O", "O", "O", "B", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]

print(sentence.split(" "))
print(len(sentence.split(" ")))
print(len(labels))
