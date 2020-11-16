# Session-Based-Recommendation-System
A session based recommendation system using Graph Neural Networks
Graph Neural Networks (GNNs)are designed for generating representations for graphs.

For the session-based recommendation,we first construct directed graphs from historical session sequences.

Based on the session graph, GNN is capable of capturing transitions of items and generating accurate item embedding vectors correspondingly, which are difficult to be revealed by the conventional sequential methods, like Markov Chain-based and RNN- based methods.

The proposed SR-GNN constructs more reliable session representations and the next-click item can be inferred.

All session sequences are modeled as directed session graphs, where each session sequence can be treated as a subgraph.

Then, each session graph is proceeded successively and the latent vectors for all nodes involved in each graph can be obtained through gated graph neural net- works.

Represent each session as a composition of the global preference and the current interest of the user in that session, where these global and local session embedding vectors are both composed by the latent vectors of nodes.

For each session, we predict the probability of each item to be the next click.

Incorporating target-attention embedding to sequential recommendation models gave a significant improvement in the recommendations so it seemed intuitive that it will give similar results when implemented on a GNN based recommendation system.

#Dataset Used
The diginetica dataset is used for this project(http://cikm2016.cs.iupui.edu/cikm-cup)

#Paper referenced
"TAGNN: Target Attentive Graph Neural Networks for Session-based Recommendation"

