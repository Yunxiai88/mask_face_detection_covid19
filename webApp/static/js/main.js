(function () {
    "use strict";

    var page = {

        ready: function () {

            var formdata = {};
            $("#processresultdiv").hide();

            // base upload function
            $('#uploadImage').fileinput({
                uploadUrl: '/uploadimage',
                theme : 'explorer-fas',
                uploadAsync: false,
                showUpload: false,
                showRemove :true,
                showPreview: true,
                showCancel:true,
                showCaption: true,
                maxFileCount: 1,
                allowedFileExtensions: ['jpg', 'png'],
                uploadExtraData: function(previewId, index) {
                    return formdata
                },
                browseClass: "btn btn-primary ",
                dropZoneEnabled: true,
                dropZoneTitle: 'Drag file here！',
            });

            // image encoding upload button
            $(".btn-upload-3").on("click", function() {
                var username = $("#username").val();
                if(!username) {
                    alert('Please Enter Name.');
                    return;
                }

                formdata = {
                  "username": $("#username").val()
                }

                $("#uploadImage").fileinput('upload');
            });

            // image encoding clear button
            $(".btn-reset-3").on("click", function() {
                $("#username").val('');
                $("#uploadImage").fileinput('clear');
            });

            // call back function for upload Image file
            $('#uploadImage').on('fileuploaded', function(event, data, previewId, index) {
                $("#username").val('');
            });

            $('#uploadFile').fileinput({
                uploadUrl: '/uploadfile',
                theme : 'explorer-fas',
                uploadAsync: false,
                showUpload: false,
                showRemove :true,
                showPreview: true,
                showCancel:true,
                showCaption: true,
                allowedFileExtensions: ['jpg', 'png', 'mp4', 'avi', 'mov'],
                maxFileSize : 153600,
                maxFileCount : 1,
                browseClass: "btn btn-primary ",
                dropZoneEnabled: true,
                dropZoneTitle: 'Drag file here！'
            });

            // image process upload button
            $(".btn-uploadfile-3").on("click", function() {
                $("#uploadFile").fileinput('upload');
            });

            // image process clear button
            $(".btn-resetfile-3").on("click", function() {
                $("#uploadFile").fileinput('clear');
            });

            // call back function for upload file
            $('#uploadFile').on('filebatchuploadsuccess', function(event, data, previewId, index) {
                var url = "/download/"+data.response.filename;
                $("#downloadbtn").attr("href", url);
                $("#processresultdiv").show();
            });

            //pending

        }
    }

    $(document).ready(page.ready);

})();