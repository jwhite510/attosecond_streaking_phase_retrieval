import numpy as np
import matplotlib.pyplot as plt


def draw_nodes(pos_x, yvals, axis):
    for ypos in yvals:
        axis.plot(pos_x, ypos, marker="o", color="black", markersize=10.0)

def connect_conv(layer1, layer2, axis):
    for i, node2 in enumerate(layer2[1]):
        for node1 in layer1[1][i:i+3]:
            ax.plot([layer2[0], layer1[0]], [node2, node1], color="black", linewidth=1.0)

def dense_layer(layer1, layer2):
    for node2 in layer2[1]:
        for node1 in layer1[1]:
            ax.plot([layer1[0],layer2[0]], [node1, node2], color="black", linewidth=1.0)

sc = 1.5
fig, ax = plt.subplots(figsize=(sc*7,sc*4))
ax.set_xlim(0,0.7)
ax.set_ylim(0.1,0.9)



# draw the input positions
input_pos_y = np.linspace(0.2, 0.8, 15)
draw_nodes(pos_x=0.1, yvals=input_pos_y, axis=ax)

# conv layer 1
conv_layer1 = input_pos_y[1:-1] 
draw_nodes(pos_x=0.2, yvals=conv_layer1, axis=ax)
connect_conv(layer1=(0.1, input_pos_y), layer2=(0.2, conv_layer1), axis=ax)

# conv layer 2
conv_layer2 = conv_layer1[1:-1] 
draw_nodes(pos_x=0.3, yvals=conv_layer2, axis=ax)
connect_conv(layer1=(0.2, conv_layer1), layer2=(0.3, conv_layer2), axis=ax)

# conv layer 3
conv_layer3 = conv_layer2[1:-1] 
draw_nodes(pos_x=0.4, yvals=conv_layer3, axis=ax)
connect_conv(layer1=(0.3, conv_layer2), layer2=(0.4, conv_layer3), axis=ax)

# dense layer 1
dense1 = conv_layer3 
draw_nodes(pos_x=0.5, yvals=dense1, axis=ax)
dense_layer(layer1=(0.4, conv_layer3), layer2=(0.5, dense1))

# dense layer 2
dense2 = dense1[1:-1] 
draw_nodes(pos_x=0.6, yvals=dense2, axis=ax)
dense_layer(layer1=(0.5, dense1), layer2=(0.6, dense2))


# ax.arrow(0.05, 0.5, 0.025, 0.0, width=0.1, head_width=0.3, head_length=0.3)
for output_node in dense2:
    ax.arrow(0.6, output_node, 0.04, 0.0, width=0.005, head_width=0.025, head_length=0.01, 
            color="black")



plt.savefig("./graph.png")

