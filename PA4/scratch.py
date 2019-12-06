import numpy as np
x = np.random.randint(10, size=(3, 4))
print(x)
print(x.shape[1])

prob = 0.3


# indices = np.random.choice(np.arange(myarray.shape[1]), replace=False,
#                            size=int(myarray.shape[1] * prob))
# print(indices)
# myarray[indices] = 0

i,j = np.nonzero(x)
print(i,j)

ix = np.random.choice(len(i), int(np.floor(0.2 * len(i))), replace=False)
print(ix)

x[i[ix], j[ix]]= 0

print(x)



#for sentence store labels and embeddings in list
# tokens contains vector of 400 dimensions for each label
labels1 = []
tokens1 = []
for i in sentence1:
    if i in e:
        labels1.append(i)
        tokens1.append(e[i])
    else:
        print i

#Plot values
x = []
y = []


for value in new_values:
    x.append(value[0])
    y.append(value[1])


plt.figure(figsize=(10, 10))

for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i],
                 xy=(x[i], y[i]),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')


plt.show()