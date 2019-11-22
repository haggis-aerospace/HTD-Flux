
struct InceptionBlock
    path_1
    path_2
    path_3
    path_4
  end

@functor InceptionBlock

function InceptionBlock(in_chs, chs_1x1, chs_3x3_reduce, chs_3x3, chs_5x5_reduce, chs_5x5, pool_proj)
  path_1 = Conv((1, 1), in_chs=>chs_1x1, relu)

  path_2 = (Conv((1, 1), in_chs=>chs_3x3_reduce, relu),
            Conv((3, 3), chs_3x3_reduce=>chs_3x3, relu, pad = (1, 1)))

  path_3 = (Conv((1, 1), in_chs=>chs_5x5_reduce, relu),
            Conv((5, 5), chs_5x5_reduce=>chs_5x5, relu, pad = (2, 2)))

  path_4 = (MaxPool((3,3), stride = (1, 1), pad = (1, 1)),
            Conv((1, 1), in_chs=>pool_proj, relu))

  InceptionBlock(path_1, path_2, path_3, path_4)
end

function (m::InceptionBlock)(x)
  cat(m.path_1(x), m.path_2[2](m.path_2[1](x)), m.path_3[2](m.path_3[1](x)), m.path_4[2](m.path_4[1](x)), dims = 3)
end
