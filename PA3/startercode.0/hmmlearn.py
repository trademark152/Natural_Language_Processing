import sys
model_file="hmmmodel.txt"
output_file="hmmoutput.txt"


if __name__=="__main__":
    train_file = sys.argv[1]
    print(train_file)
    open(model_file, "w")
