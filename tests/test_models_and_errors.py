from inference_server.schemas.openai import (
    ErrorDetail,
    ErrorResponse,
    Model,
    ModelList,
)


def test_model_list_serializes_to_openai_shape() -> None:
    # Arrange
    expected_payload = {
        "object": "list",
        "data": [
            {
                "id": "qwen2.5-0.5b-instruct",
                "object": "model",
                "created": 1752710400,
                "owned_by": "inference-server"
            }
        ]
    }

    model = Model(
        id="qwen2.5-0.5b-instruct",
        created=1752710400,
        owned_by="inference-server"
    )
    model_list = ModelList(data=[model])


    # Act
    payload = model_list.model_dump()


    # Assert
    assert payload == expected_payload


def test_error_detail_defaults_optional_fields_when_omitted() -> None:
    # Arrange
    payload = {
        "message": "Something went wrong.",
        "type": "server_error"
    }


    # Act
    detail = ErrorDetail.model_validate(payload)


    # Assert
    assert detail.param is None
    assert detail.code is None


def test_error_response_serializes_to_openai_shape() -> None:
    # Arrange
    expected_payload = {
        "error": {
            "message": "The model `missing-model` does not exist.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found"
        }
    }

    detail = ErrorDetail(
        message="The model `missing-model` does not exist.",
        type="invalid_request_error",
        param="model",
        code="model_not_found"
    )
    error_response = ErrorResponse(error=detail)


    # Act
    payload = error_response.model_dump()


    # Assert
    assert payload == expected_payload
