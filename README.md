# MLP Name Generator

A simple character-level name generator using a Multi-Layer Perceptron implemented in JAX.

This project is leading to a larger project for the next days. 

 ## Output example

Example training progress and generated names:      

      0/ 200000: 3.2973
            .
            .
            .
            .
    199000/ 200000: 2.4073

 Generating sample names: 

    $ 
    en.
    eta.
    aemoovqhgyynodpipzzp.
    vvtfatmbhagcsxjnuwjt.
    tjtewjrmpsnqjwlpjvyu.
    qvcdvsm.
    qxhoxvbvahtfqcrkdecw.
    qr.
    jyybe.
    pubi.
    jhzeyegk.
    tohzrcdmvobbtxdwnhkp.
    ynqjnwrinkihhtetkhlm.
    kllsuqqkh.
    pxumeaojqrjbuzggzbki.
    kslezfq.
    jfbsqmupq.
    vxk.
    hxopggfwmytzf.
    ```

You can see that the names generated are just random letters, that's where come the Transformer, In the jaxGPT repo, I implemented GPT like model, your exercise is to train it on names-dataset.txt, you'll notice that the model is generating real names (a name you can't use LOl!) 
## Overview
This code implements a character-level name generator using a Multi-Layer Perceptron (MLP) neural network. It learns patterns from a dataset of names and can generate new, similar-sounding names.

## References 
- Bengio et al. 2003
- Karpathy's NanoGPT    



