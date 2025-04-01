# fast_lio-sam_loop
Fast lio with loop closing function. the Transform between map coordinate to Odom coordinate is maintained and used to correct the FAST-LIO pose to the map system, providing initial pose-graph. Therefore, Fast-lio2's ikd tree does not require reconstruction, ensuring that the front-end can always run efficiently.
