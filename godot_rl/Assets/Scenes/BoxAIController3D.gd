extends AIController3D


var move = Vector2.ZERO
@onready var character_body_3d = $".."
@onready var area_3d = $"../../Area3D"

func get_obs() -> Dictionary:
	var obs := [
		character_body_3d.position.x,
		character_body_3d.position.z,
		area_3d.position.x,
		area_3d.position.z,
	]
	return {"obs": obs}

func get_reward() -> float:	
	return reward
	
func get_action_space() -> Dictionary:
	return {
		"move" : {
			"size": 2,
			"action_type": "continuous"
		},
	}
	
func reset():
	area_3d.position = Vector3(randi_range(-5, 5), 0, randi_range(-5, 5))
	character_body_3d.position = Vector3(0, 2, 0)
	super.reset()
	
func set_action(action) -> void:	
	move.x = action["move"][0]
	move.y = action["move"][1]
