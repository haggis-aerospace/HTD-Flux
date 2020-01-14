using Images

path_to_tutorial = ""
directory = "PennFudanPed/PNGImages"

global k=0
for filename in readdir(directory)
    img = load("$directory/$filename")
    rows, cols = size(img)
    for iter in 0:10
        global k += 1
        i = rand(1:rows-128)
        j = rand(1:cols-64)
        save("not_humans/$k.jpg", img[i:i+128, j:j+64])
    end
end
