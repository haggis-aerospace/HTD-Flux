using Images, ImageFeatures

path_to_tutorial = ""
pos_examples = "tutorial/humans/"
neg_examples = "tutorial/not_humans/"

n_pos = length(readdir(pos_examples))   # number of positive training examples
n_neg = length(readdir(neg_examples))   # number of negative training examples
n = n_pos + n_neg;                      # number of training examples 

data = Array{Float64}(undef, 3780, n);  # Array to store HOG descriptor of each image. Each image in our training data has size 128x64 and so has a 3780 length
labels = Vector{Int}(undef, n);         # Vector to store label (1=human, 0=not human) of each image.

for (i, file) in enumerate([readdir(pos_examples); readdir(neg_examples)])
    filename = "$(i <= n_pos ? pos_examples : neg_examples )/$file"
    img = load(filename)
    data[:, i] = create_descriptor(img, HOG())
    labels[i] = (i <= n_pos ? 1 : 0)
end;