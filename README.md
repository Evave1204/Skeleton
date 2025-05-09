### Project Overview

This project focuses on **skeletonizing the vascular tree** and extracting key structural information, including branch lengths, radii, angles between branches, and hierarchical branch levels.

### Functionality

- **Lengths and radii** are extracted using a skeletonization library, with additional post-processing to ensure precision.
- **Branch angles and hierarchical levels** are computed and visualized as part of the pipeline.

### Visualizations

#### Original Vascular Tree  
The image below shows the original vascular tree used in the experiment, extracted from a mouse tissue slice.
<img src="images/Original_Tree.png" alt="Original Tree" width="800"/>

#### Skeletonization Result  
This image illustrates the result of applying the skeletonization algorithm to the vascular tree.
<img src="images/Skeleton.png" alt="Skeleton" width="800"/>

#### Length and Radius Extraction  
Here, each node's radius is visualized, demonstrating how they collectively reconstruct the shape of the original tree.
<img src="images/Tree_Info.png" alt="Tree Info" width="800"/>

#### Branch Level Assignment  
This visualization shows how the skeletonized tree is hierarchically segmented using a river-ordering method. All extracted information is ultimately stored in a structured tree format.
<img src="images/Level_assignment.png" alt="Branch Levels" width="800"/>




