"""Sample RESTful Framework."""
from sanic import Blueprint
from sanic.response import json
from sanic_openapi import doc

# NOTE: The URL Prefix for your backend has to be the name of the backend
blueprint = Blueprint("datatrail", url_prefix="/datatrail")


@doc.summary("Hello from datatrail!")
@blueprint.get("/")
async def hello(request):
    """Sample RESTful API.

    Parameters
    ----------
    request : sanic.request
        Request object from sanic app

    Returns
    -------
    sanic.json
        HTTP returning byte array json
    """
    return json("Hello from datatrail 🦧")
