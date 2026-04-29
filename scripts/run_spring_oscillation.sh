echo "Reproducing all demos and videos, this may take several hours"

# time_step
SAVE_NAME="spring_oscillate_timestep"
python -m pygoo.app \
  --scene-json data/spring_oscillate/orig.json \
  --target-video data/spring_oscillate/orig.avi \
  --log_dir logs/$SAVE_NAME \
  --set time_step=0.0003 \
  --freeze-params mass,spring_k \
  --optimize-params time_step \
  --iterations 20 \
  --optimize-start

python plot_optimization.py logs/$SAVE_NAME/optim_log.csv
mkdir logs/$SAVE_NAME/frames
python make_timelapse.py --csv_path logs/$SAVE_NAME/optim_log.csv --output logs/$SAVE_NAME/timelapse.mp4 --target-video data/spring_oscillate/orig.avi --target-json data/spring_oscillate/orig.json --zoom 3.0


# gravity_g
SAVE_NAME="spring_oscillate_gravity_g"
python -m pygoo.app \
  --scene-json data/spring_oscillate/orig.json \
  --target-video data/spring_oscillate/orig.avi \
  --log_dir logs/$SAVE_NAME \
  --set gravity_g=-8.0 \
  --freeze-params mass,spring_k \
  --optimize-params gravity_g \
  --iterations 20 \
  --optimize-start

python plot_optimization.py logs/$SAVE_NAME/optim_log.csv
mkdir logs/$SAVE_NAME/frames
python make_timelapse.py --csv_path logs/$SAVE_NAME/optim_log.csv --output logs/$SAVE_NAME/timelapse.mp4 --target-video data/spring_oscillate/orig.avi --target-json data/spring_oscillate/orig.json --zoom 3.0


# damping_stiffness
SAVE_NAME="spring_oscillate_damping_stiffness"
python -m pygoo.app \
  --scene-json data/spring_oscillate/orig.json \
  --target-video data/spring_oscillate/orig.avi \
  --log_dir logs/$SAVE_NAME \
  --set damping_stiffness=5.0 \
  --freeze-params mass,spring_k \
  --optimize-params damping_stiffness \
  --iterations 20 \
  --optimize-start

python plot_optimization.py logs/$SAVE_NAME/optim_log.csv
mkdir logs/$SAVE_NAME/frames
python make_timelapse.py --csv_path logs/$SAVE_NAME/optim_log.csv --output logs/$SAVE_NAME/timelapse.mp4 --target-video data/spring_oscillate/orig.avi --target-json data/spring_oscillate/orig.json --zoom 3.0

# mass
SAVE_NAME="spring_oscillate_mass"
python -m pygoo.app \
  --scene-json data/spring_oscillate/orig.json \
  --target-video data/spring_oscillate/orig.avi \
  --log_dir logs/$SAVE_NAME \
  --set particle.2.mass=1.5 \
  --freeze-params spring_k \
  --iterations 10 \
  --optimize-start

python plot_optimization.py logs/$SAVE_NAME/optim_log.csv
mkdir logs/$SAVE_NAME/frames
python make_timelapse.py --csv_path logs/$SAVE_NAME/optim_log.csv --output logs/$SAVE_NAME/timelapse.mp4 --target-video data/spring_oscillate/orig.avi --target-json data/spring_oscillate/orig.json --zoom 3.0


# spring_k
SAVE_NAME="spring_oscillate_spring_k"
python -m pygoo.app \
  --scene-json data/spring_oscillate/orig.json \
  --target-video data/spring_oscillate/orig.avi \
  --log_dir logs/$SAVE_NAME \
  --set edge.1.k=400 \
  --freeze-params mass \
  --iterations 20 \
  --optimize-start

python plot_optimization.py logs/$SAVE_NAME/optim_log.csv
mkdir logs/$SAVE_NAME/frames
python make_timelapse.py --csv_path logs/$SAVE_NAME/optim_log.csv --output logs/$SAVE_NAME/timelapse.mp4 --target-video data/spring_oscillate/orig.avi --target-json data/spring_oscillate/orig.json --zoom 3.0

# floor_bounce
SAVE_NAME="floor_bounce_floor_bounce"
python -m pygoo.app \
  --scene-json data/floor_bounce/orig.json \
  --target-video data/floor_bounce/orig.avi \
  --log_dir logs/$SAVE_NAME \
  --set floor_bounce=0.25 \
  --freeze-params mass,spring_k \
  --optimize-params floor_bounce \
  --iterations 20 \
  --optimize-start

python plot_optimization.py logs/$SAVE_NAME/optim_log.csv
mkdir logs/$SAVE_NAME/frames
python make_timelapse.py --csv_path logs/$SAVE_NAME/optim_log.csv --output logs/$SAVE_NAME/timelapse.mp4 --target-video data/floor_bounce/orig.avi --target-json data/floor_bounce/orig.json --zoom 1.5


# random
# observable-json will just use default values for masses and stiffnesses
SAVE_NAME="random_mass_spring_k"
python -m pygoo.app \
  --observable-json data/random/orig.json \
  --target-video data/random/orig.avi \
  --log_dir logs/$SAVE_NAME \
  --iterations 100 \
  --optimize-start \
  --curriculum-advance-threshold 0.000005

python plot_optimization.py logs/$SAVE_NAME/optim_log.csv
mkdir logs/$SAVE_NAME/frames
python make_timelapse.py --csv_path logs/$SAVE_NAME/optim_log.csv --output logs/$SAVE_NAME/timelapse.mp4 --target-video data/random/orig.avi --target-json data/random/orig.json --zoom 1.0

