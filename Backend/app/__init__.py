def create_app():
	from app.create_app import create_app as _create_app

	return _create_app()


__all__ = ["create_app"]
