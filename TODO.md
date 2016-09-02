1) Test the batch predict and prepare batches pipeline
    - looks like it is in good shape, still needs polishing
    - So we learned some things from working with the data.
        1) we dont always want a generator and handling the shape of the batches is non-trival
        2) Sometimes we do want a generator (print as we go type thing)
        3) Sometimes we dont want to do anything until the data is done.
        4) Figure out how to smartly handle the batches wrt generator vs sending all at once
        5) We dont always care about the name of the sequences being classified and it is annoyihng dealing with that. Perhaps override the function to correct for that. (as an option)

2) Work on clustering higher dimensionality thing   
    - Done!

2) Finish filling out the main train/test package

3) Fill out the post-processing/evaluation metrics
    - Shuffling
    - Pointwise
    - Overtime -- try modifying the model and loading weights but return - sequences = true, just need to apply the LR layer to each output
    

4) add an option for how to return stuff from load_fasta, sometimes the extra information is annoying
