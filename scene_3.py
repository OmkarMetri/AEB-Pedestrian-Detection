import carla
import random
import numpy as np
import time
from yolov8_detect_pedestrians import YOLOv8PedestrianDetector


class Config:
    def __init__(self, town="Town10HD", ego_vehicle_filter="vehicle.toyota.prius"):
        self.server = "127.0.0.1"
        self.port = 2000
        self.timeout = 100.0
        self.display_size = 400
        self.obs_range = 50
        self.d_behind = 12
        self.fov = 150
        self.town = town
        self.ego_vehicle_filter = ego_vehicle_filter
        self.walker_speed = '1.0'
        self.scene = 'images/scene3'
        self.model = YOLOv8PedestrianDetector()


def create_vehicle_blueprint(bp_library, actor_filter, color=None):
    blueprints = bp_library.filter(actor_filter)
    blueprints = [
        x for x in blueprints if int(x.get_attribute("number_of_wheels")) == 4
    ]
    bp = random.choice(blueprints)
    if bp.has_attribute("color"):
        if not color:
            color = random.choice(
                bp.get_attribute("color").recommended_values)
        bp.set_attribute("color", color)
    return bp


def create_walker_blueprint(bp_library, actor_filter):
    blueprints = bp_library.filter(actor_filter)
    bp = random.choice(blueprints)
    if bp.has_attribute('is_invincible'):
        bp.set_attribute('is_invincible', 'false')
    return bp


def spawn_obstacle(world, blueprint_library, location):
    obstacle_bp = blueprint_library.find('static.prop.trafficcone01')
    obstacle_transform = carla.Transform(location)
    obstacle_actor = world.spawn_actor(obstacle_bp, obstacle_transform)
    return obstacle_actor


def camera_callback(conf, image, vehicle, walker, brake_distance=15.0):
    if int(image.frame) % 3 != 0:
        return

    array = np.reshape(image.raw_data, (image.height, image.width, 4))
    array_3channel = array[:, :, :3]
    pedestrian_detected = conf.model.detect(array_3channel)

    vehicle_location = vehicle.get_location()
    pedestrian_location = walker.get_location()
    distance_to_pedestrian = vehicle_location.distance(pedestrian_location)

    if pedestrian_detected:
        print(f"Pedestrian visible at frame {image.frame}. Distance: {distance_to_pedestrian:.2f} m")
        image.save_to_disk(f'{conf.scene}/{image.frame:06d}-visible.png')
        if distance_to_pedestrian < brake_distance:
            print("Braking for pedestrian...")
            custom_autopilot(vehicle, auto=False, brake=True)
    else:
        print(f"Pedestrian occluded at frame {image.frame}.")
        image.save_to_disk(f'{conf.scene}/{image.frame:06d}-occluded.png')


def custom_autopilot(vehicle, auto, brake):
    vehicle.set_autopilot(auto)
    if brake:
        control = carla.VehicleControl()
        control.brake = 1.0
        vehicle.apply_control(control)


def main():
    try:
        # Connect to the Carla server
        conf = Config()
        client = carla.Client(conf.server, conf.port)
        client.set_timeout(conf.timeout)
        world = client.load_world(conf.town)
        blueprint_library = world.get_blueprint_library()
        print("Carla server connected!")

        # traffic manager
        traffic_manager = client.get_trafficmanager()

        # Set weather
        world.set_weather(carla.WeatherParameters.ClearNoon)

        # Spawn the ego vehicle
        ego_vehicle_bp = create_vehicle_blueprint(blueprint_library, conf.ego_vehicle_filter, color="49,8,8")
        ego_vehicle_spawn_point = carla.Transform(
            carla.Location(x=-68.735168, y=129.303848, z=0.600000),
            carla.Rotation(pitch=0.000000, yaw=-167.127060, roll=0.000000)
        )
        ego_vehicle = world.spawn_actor(ego_vehicle_bp, ego_vehicle_spawn_point)
        print("Spawned ego vehicle!")

        # Camera settings
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.5), carla.Rotation(pitch=0, yaw=0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

        # Spawn the pedestrian
        src_crosswalk_location = carla.Location(x=-89.514999-7, y=31.997713, z=1.0)
        dst_crosswalk_location = carla.Location(x=-97.911476-22,  y=38.460583, z=1.0)

        walker_bp = create_walker_blueprint(blueprint_library, 'walker.pedestrian.0026')
        walker_bp.set_attribute('speed', conf.walker_speed)
        walker = client.apply_batch_sync([carla.command.SpawnActor(walker_bp, carla.Transform(src_crosswalk_location))], True)
        walker_id = walker[0].actor_id
        walker_actor = world.get_actor(walker_id)

        walker_controller_bp = blueprint_library.find('controller.ai.walker')
        walker_controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker_actor)
        walker_controller.start()
        walker_controller.go_to_location(dst_crosswalk_location)
        print(f"Spawned walker with ID {walker_id}")

        # Spawn obstacle
        obstacles = []
        obstacle_locations = [carla.Location(x=-97.911476-3,  y=40.460583, z=0.0),
                            carla.Location(x=-97.911476-3,  y=40.460583, z=0.5),
                            carla.Location(x=-97.911476-3,  y=40.460583, z=1.0),
                            carla.Location(x=-97.911476-3,  y=40.460583, z=1.5)]
        for obstacle_location in obstacle_locations:
            obstacles.append(spawn_obstacle(world, blueprint_library, obstacle_location))

        # Set autopilot and camera listener
        camera.listen(lambda image: camera_callback(conf, image, ego_vehicle, walker_actor))
        ego_vehicle.set_autopilot(True)
        traffic_manager.ignore_lights_percentage(ego_vehicle, 100)

        while True:
            world.tick()

    except KeyboardInterrupt:
        pass

    finally:
        # Cleanup
        if 'ego_vehicle' in locals():
            camera.destroy()
            ego_vehicle.destroy()
        if 'walker_controller' in locals():
            walker_controller.stop()
            walker_controller.destroy()
        if 'walker_actor' in locals():
            walker_actor.destroy()
        if 'obstacle_actor' in locals():
            for obstacle_actor in obstacles:
                obstacle_actor.destroy()
        print("Cleaned up and destroyed actors.")


if __name__ == '__main__':
    main()
