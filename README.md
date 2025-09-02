# WebCoach

## Setup: browser-use and docker

First clone the repository (or fetch it if you've already cloned it):

```
git clone https://github.com/browser-use/browser-use.git
cd browser-use

git fetch --tags

git checkout tags/0.7.1 -b my-0.7.1-branch
```

Then install docker image on this machine:

First change one line in `/home/genglin/browser-use/docker/build-base-images.sh`

```
$build_cmd $tag_args $build_args -f $dockerfile ..
```

Then follow `/home/genglin/browser-use/docker/README.md` for docker image installation

## Setup: running local models

Follow the scripts under `/home/genglin/scripts_gl/start_LLM_inference/ASG`. SGLang provides their own docker image already.

## Running the Benchmark

First make sure you configurate the experiments here using `/home/genglin/scripts_gl/run_benchmark/config.yaml`

Then follow `/home/genglin/scripts_gl/run_benchmark/run_webvoyager.sh`

## Evaluation

Run `/home/genglin/scripts_gl/evaluation/eval_webvoyager_results.py` to generate table, metrics include SR, Succ/Total, Avg_time, Avg_num_steps.