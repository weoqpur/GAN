# pipeline

# generator
# input -> Conv(k9n64s1)-> PReLU-> (Conv(k3n64s1)-> BN-> PReLU-> Conv(k3n64s1)-> BN) * 5->
# Conv(k3n64s1)-> BN-> Elementwise Sum->
# (Conv(k3n256s1)-> PixelShuffler x2-> PReLU) *2-> Conv(k9n3s1) -> output

# discriminator
# input -> Conv(k3n64s1)-> Leaky ReLU-> (Conv(k3n64s2)-> BN-> Leaky ReLU)->
# (Conv(k3n128s1)-> BN-> Leaky ReLU)-> (Conv(k3n128s2)-> BN-> Leaky ReLU)->
# (Conv(k3n256s1)-> BN-> Leaky ReLU)-> (Conv(k3n256s2)-> BN-> Leaky ReLU)->
# (Conv(k3n512s1)-> BN-> Leaky ReLU)-> (Conv(k3n512s2)-> BN-> Leaky ReLU)->
# Dense (1024)-> Leaky ReLU-> Dense (1)-> Sigmoid -> 1 or 0
