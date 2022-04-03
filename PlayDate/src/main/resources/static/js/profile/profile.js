"use strict";

$(document).ready(function () {
    $('#uploadButton').prop('disabled', true);

    $(document).on('change', '.btn-file :file', function () {
        var input = $(this),
            label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
        input.trigger('fileselect', [label]);
    });

    $('.btn-file :file').on('fileselect', function (event, label) {

        var input = $('#current-file'),
            log = label;

        if (input.length) {
            input.val(log);
        } else {
            if (log) alert(log);
        }

    });

    function readProfileImage(input) {
        if (input.files && input.files[0]) {
            $('#imageError').text("");
            $('#uploadButton').prop('disabled', true);

            let reader = new FileReader();

            if (input.files[0].size > 10485760) {
                $('#img-upload').attr('src', "");
                $('#imageError').text("Image too big");
                return;
            }

            let image = new Image();

            reader.onload = function (e) {
                image.onload = function () {
                    $('#img-upload').attr('src', e.target.result);
                    $('#uploadButton').prop('disabled', false);
                };

                image.onerror = function () {
                    $('#img-upload').attr('src', "");
                    $('#imageError').text("Invalid file");
                };

                image.src = e.target.result;
            };

            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imgInp").change(function () {
        readProfileImage(this);
    });
});

let followRequestLoading = false;
function followUnfollow(e, userId) {
    e.preventDefault();
    if (followRequestLoading) return;

    followRequestLoading = true;

    let selector = '#follow-unfollow-'+userId;

    if ($(selector).hasClass('fa-user-plus')) {
        $.post(
            `/user/${userId}/follow`,
            {},
            function (success) {
                $(selector).removeClass('fa-user-plus').addClass('fa-user-minus');
                $(selector).removeClass('hoverable');
                followRequestLoading = false;
            }
        )
    } else {
        $.post(
            `/user/${userId}/unfollow`,
            {},
            function (success) {
                $(selector).removeClass('fa-user-minus').addClass('fa-user-plus');
                $(selector).addClass('hoverable');
                followRequestLoading = false;
            }
        )
    }
}

let banRequestLoading = false;
function banUnban(e, userId) {
    e.preventDefault();
    if (banRequestLoading) return;

    banRequestLoading = true;

    let selector = '#ban-'+userId;

    if ($(selector).hasClass('hoverable')) {
        $.post(
            `/user/${userId}/ban`,
            {},
            function (success) {
                $(selector).removeClass('hoverable');
                banRequestLoading = false;
            }
        )
    } else {
        $.post(
            `/user/${userId}/unban`,
            {},
            function (success) {
                $(selector).addClass('hoverable');
                banRequestLoading = false;
            }
        )
    }
}

function editProfile() {
    let firstName = document.getElementById('firstName').value;
    let lastName = document.getElementById('lastName').value;
    let phone = document.getElementById('phone').value;
    let country = document.getElementById('country').value;
    let city = document.getElementById('city').value;
    let about = document.getElementById('about').value;
    let oldEmail = document.getElementById('oldEmail').value;
    let newEmail = document.getElementById('newEmail').value;
    let oldPassword = document.getElementById('oldPassword').value;
    let newPassword = document.getElementById('newPassword').value;
    let confirmPassword = document.getElementById('confirmPassword').value;

    let editProfileFormData = {
        firstName,
        lastName,
        phone,
        country,
        city,
        about,
        oldEmail,
        newEmail,
        oldPassword,
        newPassword,
        confirmPassword
    };

    checkEditProfileFormData(editProfileFormData);
}

function checkEditProfileFormData(editProfileFormData) {
    resetModal('#editModal', false);

    let editProfileInfo;

    $.ajax({
        type: 'POST',
        url: '/profile/edit/information',
        data: JSON.stringify(editProfileFormData),
        contentType : 'application/json',
        dataType : 'json',
        async: false,
        success: function(info) {
            editProfileInfo = info;
        }
    });

    let wasError = false;
    if (editProfileFormData.firstName) {
        if (editProfileFormData.firstName.length < 2 || editProfileFormData.firstName.length > 30) {
            appendError('Length must be between 2 and 30 characters', 'first-name-form-group');
            wasError = true;
        }
    } else {
        appendError('Length must be between 2 and 30 characters', 'first-name-form-group');
        wasError = true;
    }

    if (editProfileFormData.lastName) {
        if (editProfileFormData.lastName.length < 2 || editProfileFormData.lastName.length > 30) {
            appendError('Length must be between 2 and 30 characters', 'last-name-form-group');
            wasError = true;
        }
    } else {
        appendError('Length must be between 2 and 30 characters', 'last-name-form-group');
        wasError = true;
    }

    if (!isPhoneValid(editProfileFormData.phone)) {
        appendError('Invalid phone number format', 'phone-form-group');
        wasError = true;
    }

    console.debug(editProfileInfo);
    console.debug(editProfileFormData);

    if (editProfileFormData.oldEmail) {
       let email = editProfileInfo.email;

        if (email !== editProfileFormData.oldEmail) {
            appendError('Incorrect email', 'old-email-form-group');
            wasError = true;
        } else if (!editProfileFormData.newEmail) {
            appendError('Please enter a new email address', 'new-email-form-group');
            wasError = true;
        } else if (!isEmailValid(editProfileFormData.newEmail)) {
            appendError('Please enter a valid email address', 'new-email-form-group');
            wasError = true;
        } else {
            if (editProfileInfo.emailUsed) {
                appendError('Email already in use', 'new-email-form-group');
                wasError = true;
            }
        }
    } else {
        if (editProfileFormData.newEmail) {
            appendError('Please enter your old email first', 'new-email-form-group');
            wasError = true;
        }
    }

    if (editProfileFormData.oldPassword) {

        if (!editProfileInfo.passwordValid) {
            appendError('Please enter your current password', 'old-password-form-group');
            wasError = true;
        } else {
            if (!editProfileFormData.newPassword) {
                appendError('Please enter the desired password', 'new-password-form-group');
                wasError = true;
            } else {
                if (editProfileFormData.confirmPassword !== editProfileFormData.newPassword) {
                    appendError('Passwords don\'t match', 'confirm-password-form-group');
                    wasError = true;
                }
            }
        }
    } else {
        if (editProfileFormData.newPassword) {
            appendError('Please enter your old password', 'new-password-form-group');
            wasError = true;
        }
        if (editProfileFormData.confirmPassword) {
            appendError('Please enter your old password', 'confirm-password-form-group');
            wasError = true;
        }
    }

    if (wasError) return;
    if (editProfileFormData.firstName === editProfileInfo.firstName &&
        editProfileFormData.lastName === editProfileInfo.lastName &&
        editProfileFormData.phone === editProfileInfo.phone &&
        editProfileFormData.country === editProfileInfo.country &&
        editProfileFormData.city === editProfileInfo.city &&
        editProfileFormData.about === editProfileInfo.about &&
        !editProfileFormData.oldEmail &&
        !editProfileFormData.newEmail &&
        !editProfileFormData.oldPassword &&
        !editProfileFormData.newPassword &&
        !editProfileFormData.confirmPassword) {
        return;
    }

    console.debug(JSON.stringify(editProfileFormData));

    $.ajax({
        type: 'POST',
        url: '/profile/edit',
        data: JSON.stringify(editProfileFormData),
        contentType : 'application/json',
        success: function () {
            appendSuccess('The profile has been edited!', 'confirm-password-form-group');
        }, error: function() {
            appendError('An error happened while updating the profile!', 'confirm-password-form-group');
        }
    });
    resetModal('#editModal', false);
    reloadPage("#editModal");
}
