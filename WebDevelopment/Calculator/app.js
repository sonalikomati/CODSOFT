$(document).ready(function () {
    let expression = ""; // Store the mathematical expression

    // Update the display values
    function updateDisplay() {
        $("#expression_display").text(expression); // Show the full expression
        try {
            // Replace custom operators and safely evaluate the expression
            let evaluatedExpression = expression
                .replace(/×/g, "*")
                .replace(/÷/g, "/")
                .replace(/%/g, "/100"); // Interpret % as divide by 100

            let result = eval(evaluatedExpression); // Evaluate the expression
            $("#val_display").text(isNaN(result) ? "" : result); // Display the result
        } catch {
            $("#val_display").text(""); // Clear the result if expression is invalid
        }
    }

    // Clear everything (AC button)
    $("#reset").click(function () {
        expression = "";
        updateDisplay();
    });

    // Backspace functionality (delete last character)
    $("#backspace").click(function () {
        expression = expression.slice(0, -1); // Remove the last character
        updateDisplay();
    });

    // Handle button clicks for numbers and operators
    $(".cal-btn").not("#reset, #backspace").click(function () {
        const value = $(this).text();

        if (value === "=") {
            try {
                // Replace custom operators and safely evaluate the expression
                expression = eval(
                    expression.replace(/×/g, "*").replace(/÷/g, "/").replace(/%/g, "/100")
                ).toString(); // Evaluate and update the expression
            } catch {
                expression = ""; // Reset expression on invalid input
                $("#val_display").text("Error");
            }
        } else {
            // Add the button value to the expression
            expression += { "×": "*", "÷": "/", "%": "%" }[value] || value;
        }

        updateDisplay();
    });

    // Initialize the display
    updateDisplay();
});
