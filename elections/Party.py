class Party:
    def __init__(self, name: str, short_name: str):
        self.name = name
        self.short_name = short_name


Republicans = Party("Republican", "rep")
Democrats = Party("Democrat", "dem")
Independents = Party("Independent", "ind")
