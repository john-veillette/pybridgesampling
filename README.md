This repository contains a `bridge_sampler` function for estimating the marginal likelihood of a PyMC model using bridge sampling, which is a prerequisite to calculating Bayes Facotrs. It is based on Junpeng Lao's [Python port](https://junpenglao.xyz/Blogs/posts/2017-11-22-Marginal_likelihood_in_PyMC3.html) of Quentin Gronau's R tutorial that accompanies his and his colleagues' [tutorial paper](https://doi.org/10.1016/j.jmp.2017.09.005), but this version is updated for compatibility with modern PyMC and ArviZ. Both Junpeng and Quentin do an excellent job explaining bridge sampling, so please refer to their work to figure out what's going on for the time being, as I have made no attempt to do so here yet.

> __Warning!__ I have not checked, but this probably doesn't work with models that contain a `pymc.Potential`. Proceed with caution if that's your situation.

> __Warning!__ This seems to work with PyMC's default sampler, but not if you use `nuts_sampler = 'numpyro' in `pymc.sample`. Not sure about other sampler options.

As evidenced by the above warnings, this is a work in progress. It reproduces numbers from Junpeng's and Quentin's tutorials (yay) so seems to be "correct," but I haven't yet stress tested the code across all reasonable PyMC use cases.
