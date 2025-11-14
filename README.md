# WebCoach

## Setup: browser-use and docker

First `cd ~` and then clone the repository (or fetch it if you've already cloned it):

```
git clone https://github.com/browser-use/browser-use.git
cd browser-use

git fetch --tags

git checkout tags/0.7.1 -b my-0.7.1-branch
```

Then install docker image on this machine:

First change one line in `~/browser-use/docker/build-base-images.sh`

```
$build_cmd $tag_args $build_args -f $dockerfile ..
```

Then follow `~/browser-use/docker/README.md` for docker image installation

## Setup: running local models

Follow the scripts under `~/start_LLM_inference`. SGLang provides their own docker image already.

## Running the Benchmark

First make sure you configurate the experiments here using `~/run_benchmark/config.yaml`

Then follow `~/run_benchmark/run_webvoyager.sh`

You can specify base_dir, model, benchmark, and coaching parameters

## Evaluation

Run `~/evaluation/eval_webvoyager_results.py` to generate table, metrics include SR, Succ/Total, Avg_time, Avg_num_steps.