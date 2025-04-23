// static/js/script.js
$(document).ready(function () {
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    $('#imageUpload').change(function () {
        const reader = new FileReader();
        reader.onload = function (e) {
            $('#imagePreview').attr('src', e.target.result);
            $('.image-section').show();
        };
        reader.readAsDataURL(this.files[0]);
    });

    $('#btn-predict').click(function () {
        const form_data = new FormData($('#upload-file')[0]);
        $(this).prop('disabled', true);
        $('.loader').show();
        $('#result').hide();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function (response) {
                $('.loader').hide();
                $('#btn-predict').prop('disabled', false);

                if (response.plant && response.condition) {
                    const text = `
                        <strong>Plant:</strong> ${response.plant}<br>
                        <strong>Condition:</strong> ${response.condition}<br>
                        <strong>Confidence:</strong> ${response.confidence}%
                    `;
                    $('#result span').html(text);
                    $('#result').show();
                } else {
                    $('#result span').html("⚠️ Unexpected response format.");
                    $('#result').show();
                }
            },
            error: function (xhr, status, error) {
                $('.loader').hide();
                $('#btn-predict').prop('disabled', false);
                console.error('Prediction error:', xhr.responseText);
                $('#result span').html("❌ Error: Unable to process prediction.");
                $('#result').show();
            }
        });
    });
});
