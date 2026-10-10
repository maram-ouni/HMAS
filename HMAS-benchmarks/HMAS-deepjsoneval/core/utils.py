def extract_task_type(task_id: str) -> str:
    """
    Extrait le type de tâche depuis l'identifiant.
    Ex: "deepjsoneval_42_patient" -> "patient"
        "deepjsoneval_7_game"     -> "game"
        "row_15"                  -> "row"
    """
    parts = task_id.split("_")
    if len(parts) >= 3:
        return parts[-1]
    return task_id.split("_")[0] if "_" in task_id else task_id
