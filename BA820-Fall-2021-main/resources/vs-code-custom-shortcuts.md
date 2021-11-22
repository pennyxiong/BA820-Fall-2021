# About

In VS Code, I create a custom keyboard shortcut so that control+enter submit the current line of my script, and goes to the next.  This is standard functionality in RStudio.

## Tools

I am able to do this via the Marketplace extension macros, details below

Name: macros
Id: geddski.macros
Description: automate repetitive actions with custom macros
Version: 1.2.1
Publisher: geddski
VS Marketplace Link: https://marketplace.visualstudio.com/items?itemName=geddski.macros

## Setup

I edited my Keybaord Shortcuts JSON file by searching for it within the Command Palette.

With the JSON file open, I have the following entry

```
[
    {
        "key": "ctrl+enter",
        "command": "macros.pythonExecSelectionAndCursorDown",
        "when": "editorTextFocus && editorLangId == 'python'"
    }
]
```

Note the `[]` that are on the outside.  It's entirely possible that you have other entries there, but above, is the foundation for what we need.

After saving the file, use the command palette to open up Settings (JSON) file.

I made sure that this entry was added:

```
    "macros": {  // Note: this requires macros extension by publisher:"geddski" 
        "pythonExecSelectionAndCursorDown": [
            "python.execSelectionInTerminal", 
            "cursorDown" 
        ]
    }
```

Note that above, the entries are comma separated, so be careful when adding the entry above.

That _should_ be it, but if you have any questions, please don't hesiate to reach out on the forums!
