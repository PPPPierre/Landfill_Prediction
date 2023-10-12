class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def __call__(self, target):
        return self.register(target)

    def register(self, target):
        def add_item(key, value):
            if not callable(value):
                raise TypeError(f"Error:{value} must be callable!")
            if key in self._dict:
                # print(f"\033[31mWarning:\033[0m {value.__name__} already exists and will be overwritten!")
                pass
            self[key] = value
            return value
        
        if callable(target):
            """
            target callable --> no call name --> use the input func name or class name as registry name
            Exemple:
                register = TestRegister()
                @resgister
                class MyClass:
                    def __init__(self) -> None:
                        ...
            """
            return add_item(target.__name__, target)
        else:
            """
            Not callable --> passed by the register name --> use the register name as the key
            Exemple:
                register = TestRegister()
                @resgister("my_class")
                class MyClass:
                    def __init__(self) -> None:
                        ...
            """
            return lambda x : add_item(target, x)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()