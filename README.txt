This repository was reset for clarity. Repository master is a sloppy programmer, so the previous git history is very
hard to understand. There were to many unrelated changes in commits and a lot of directory alterations as well.
Hopefully, things will go better with this second initial commit.

For some description of what's been worked on before the second initialization, here's brief history prior to it:

commit 09ec4f58b70cd9ef56a56316d86e46e01d8f56d1
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Aug 3 21:25:22 2016 +0600

    Commit all changes (logical and otherwise)

    This version of the repository is close to final

commit fcbdc94da2f22be9f6041032b66184227579099b
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Aug 3 21:14:07 2016 +0600

    Add gitignore to root

commit b9b3803086879736caa2a4adc98258ec12754897
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Aug 3 21:10:31 2016 +0600

    Add new XOR version

    New version that uses new layer construction interface and replicates
    PDPTool data

commit ce6733b9d269d4372b8f5c4b0dbadd51e5d44bdb
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Aug 3 03:29:16 2016 +0600

    Improve run_trainig() module

    Add several modules and improve encapsulation. Standardize printed
    output to 10 reports per training. Introduce new features (store
    hyperparams and evaluate performance). Probably there's something else
    but I forget.

commit ffa3e780ac49d6f5a86785100c4257f6da6232bb
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Aug 3 03:26:13 2016 +0600

    Refactor function inputs for clarity

commit 77709a677f3d82cc780294b169072c7b2f02b4be
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Aug 3 03:25:00 2016 +0600

    Add actual code to the modules created in previous commit

commit e875d7918a10dff96546bd3bbaa1ef6cc8ddcf5a
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Aug 3 03:19:13 2016 +0600

    Add two utility modules for (1) evaluating performance metrics and
    (2) storing hyperparameter information in the session log directory

commit 620a99c9ed4613d41b766f2e819f9f708e452674
Author: Alex <tenalexander1991@gmail.com>
Date:   Tue Aug 2 19:17:56 2016 +0600

    Add functionality to allow restoring XOR parameters for exercises

    To do next: "synch" sandbox.py with xor.py for more convenient comparison. Add performance measure to run_training()

commit 8f68877013b801dc4948793be3da3b7003fdae1a
Author: Alex <tenalexander1991@gmail.com>
Date:   Tue Aug 2 17:27:09 2016 +0600

    Add randomization features

    Add an option to permute dataset (ptrain mode) during training. Add
    seed parameter to Layer class. Weights and biases can be generated
    pseudorandmly with the seed.

commit 3a0f51830ba1f74d5ef5ac28f05ff303eb1c31d0
Author: Alex <tenalexander1991@gmail.com>
Date:   Mon Aug 1 22:29:24 2016 +0600

    Major redesign

    Add several used for training with the new construction interface

commit 2d7d5ff70d8223ae63b68d9803d0e34b645e967c
Author: Alex <tenalexander1991@gmail.com>
Date:   Mon Aug 1 15:44:40 2016 +0600

    Directory rearrangements

    Add utility directory with various utility functions. Get rid of the
    temporary ideas directory and move its contents to reference_files
    outside the main FFBP directory

commit a6a83114a07c891a8e38a123b24cf9a7594c6961
Author: Alex <tenalexander1991@gmail.com>
Date:   Sun Jul 31 17:01:07 2016 +0600

    Fix bugs to make the XOR forward prop work with new layer constructor

commit 17927482d7a2047ef45113f93614b8c62bc1d103
Author: Alex <tenalexander1991@gmail.com>
Date:   Sun Jul 31 15:31:19 2016 +0600

    Major redesign

    Remove Network Class and change Layer Class to allow simple layer constuction

commit 64788b6392195e6a77f7ad6bfe8540b6f64c38fc
Author: Alex <tenalexander1991@gmail.com>
Date:   Sun Jul 31 10:37:42 2016 +0600

    Save repo before major redesign

commit ad8d8acee190c69238bf4ea75e57f30e796793c0
Author: Alex <tenalexander1991@gmail.com>
Date:   Fri Jul 29 16:35:00 2016 +0600

    Create logdir() function for session logging

commit 6a644b7ec5a5f97081d0ac03e351ddbb0cc69a12
Author: Alex <tenalexander1991@gmail.com>
Date:   Thu Jul 28 19:05:09 2016 +0600

    Minor directory changes

commit 1090d3cd21ae3750861fc8034c53ea7ea6b48e4d
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Jul 27 21:20:26 2016 +0600

    Restructure the directory and complete the FCN.f_prop() method

commit 13e273290b4e8046f542474e7ee50e3ffafe0f8a
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Jul 27 17:27:19 2016 +0600

    Minor fix in Network.describe() method

commit e4a64b9fee4e84c08c39275d9bf4c3a3e9e109e3
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Jul 27 16:46:43 2016 +0600

    Add activation function options for method activate() in class Layer

commit 77829ffa9063c145b41a644744e99bbd68aba2f3
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Jul 27 16:36:05 2016 +0600

    Add weights_to_range() method to class Layer and tune_all_weights() method to class Network

commit 510130d981478a5761db1c4adfd26d66559d2c28
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Jul 27 16:00:49 2016 +0600

    Carry over the zero-initialization method to the Layer class

    To set all weights to zero, the Network class can call zero_all_weights()
    that will call the weights_to_zeros() method of each Layer in the Network

commit cd2f139e1eb1a37e125417bb9a88efa56638201e
Author: Alex <tenalexander1991@gmail.com>
Date:   Wed Jul 27 15:50:35 2016 +0600

    Add weights_to_zeros() method for zero-initialization of weights

commit 148549e36c6d18c3c6e5a585973e7765b15513bf
Author: Alex <tenalexander1991@gmail.com>
Date:   Tue Jul 26 20:34:21 2016 +0600

    Miniscule code change

    Assign to a local variable _name a string 'output' to pass it to activation ops

commit 1180cfd5c1e1d7611efa1f96ae6e56f607e05665
Author: Alex <tenalexander1991@gmail.com>
Date:   Tue Jul 26 20:25:11 2016 +0600

    Fix variable name attributes and define layer linear output as netinp identity

commit 033d1c25a30dc3f84f1bccb1ca1eb8a51921ebd3
Author: Alex <tenalexander1991@gmail.com>
Date:   Tue Jul 26 04:59:43 2016 +0600

    Add reference code from Tensorflow

commit 6c634a00b47d1cee7dd45562122373ea259accd1
Author: Alex <tenalexander1991@gmail.com>
Date:   Tue Jul 26 04:49:44 2016 +0600

    Add Network and Layer classes

    With these classes, user can create a network with an
    arbitrary number of layers. The created layers are
    connected (i.e weights, biases and activations are
    initialized)

commit 154b9fca8404917b7850e41d758a1a35eedcd240
Author: Alex <tenalexander1991@gmail.com>
Date:   Mon Jul 25 14:50:49 2016 +0600

    Initial commit
