from manim import *


class ServerArchitecture(Scene):
    def construct(self):
        # left group header
        path_planning = Text("Path Planning", font_size=30).to_corner(UL, buff=1)

        # Define the FastAPI server endpoints
        endpoints = VGroup(
            Text("DQN", font_size=24),
#            Text("A*", font_size=24),
            Text("RRT", font_size=24)
        ).arrange(DOWN, buff=0.5).next_to(path_planning, DOWN, buff=0.5)

        # Create boxes around each endpoint
        endpoint_boxes = VGroup(*[SurroundingRectangle(endpoint, color=BLUE, buff=0.1) for endpoint in endpoints])

        self.play(
            Write(path_planning),
            Write(endpoints),
            *[Create(box) for box in endpoint_boxes]  # Draw all boxes
            )

        # middle group header
        simulation = Text("Simulation", font_size=30).next_to(path_planning, RIGHT, buff=2)

        # Define the main function block
        main_function = Text("main()", font_size=24).next_to(simulation, DOWN, buff=0.5)
        main_box = SurroundingRectangle(main_function, color=BLUE, buff=0.1)
        self.play(
            Write(simulation),
            Write(main_function),
            *[Create(main_box)]
            )

        # Define blocks for pymavlink, ArduPilot, and QGroundControl
        stacks = VGroup(
        Text("pymavlink", font_size=16),
        Text("ArduPilot", font_size=30),
        Text("QGroundControl", font_size=16)
        ).arrange(DOWN, buff=0.2).next_to(main_function, DOWN, buff=0.5)
        stack_boxes = SurroundingRectangle(stacks, color=ORANGE, buff=0.1)

        self.play(
            Write(stacks),
            Create(Line(main_box.get_bottom(), stack_boxes.get_top())),
            Create(stack_boxes) 
            )

        # Connections and animations of QGC misson plans
        for endpoint, box in zip(endpoints, endpoint_boxes):
            line = Line(box.get_right(), main_box.get_left())
            self.play(Create(line))
            # mission plan animation
            mission_packet = Rectangle(width=0.6, height=0.4, color=RED)
            mission_text = Text("QGC", font_size=14, color=RED, slant=ITALIC, weight=ULTRAHEAVY).move_to(mission_packet.get_center())
            packet_group = VGroup(mission_packet, mission_text)
            packet_group.move_to(line.get_start())
            self.play(MoveAlongPath(packet_group, line), run_time=2)
            self.remove(packet_group)


        # right group header
        drone_flight = Text("Drone Flight", font_size=30).next_to(simulation, RIGHT, buff=2)

        # Define the Drone block
        flights = VGroup(
        Text("PixHawk", font_size=24),
        Text("MAVSDK", font_size=16),
        Text("DJI Mini 2", font_size=30)
        ).arrange(DOWN, buff=0.2).next_to(drone_flight, DOWN, buff=0.5)
        flight_box = SurroundingRectangle(flights, color=ORANGE, buff=0.1) 

        # Output from the last stack block to the Drone
        flight_line = Line(main_box.get_right(), flight_box.get_left())

        self.play(
            Write(drone_flight),
            Create(flight_line),
            Write(flights),
            Write(flight_box),
        )

        # Animation of send mission plan to drone
        mission_plan = Rectangle(width=0.6, height=0.4, color=RED)
        mission_text = Text("QGC", font_size=14, color=RED, slant=ITALIC, weight=ULTRAHEAVY).move_to(mission_plan.get_center())
        mission_group = VGroup(mission_plan, mission_text)
        mission_group.move_to(flight_line.get_start())
        self.play(MoveAlongPath(mission_group, flight_line), run_time=2)
        self.remove(mission_group)

        # Create a drone 
        drone = SVGMobject("drone.svg") 
        drone.set_color(WHITE).scale(0.5)
        drone.next_to(flight_box, DOWN, buff=0.2)

        # Animate the drone flight
        self.play(
            drone.animate.scale(0.1).move_to(3 * DOWN + 2 * LEFT),
            rate_func=there_and_back,  # flying style, adjustable 
            run_time=5  # flying speed, adjustable
        )

        # Hold the final scene
        self.wait(3)
