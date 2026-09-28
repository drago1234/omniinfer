// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#include <omni_reload.h>
#include <omni_proxy.h>
#include <sys/file.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OMNI_RELOAD_BUF_SIZE   4096
#define OMNI_RELOAD_LOCK_FILE  "/tmp/omni_reload.lock"
#define OMNI_RELOAD_CONF_MAX   (64 * 1024)

extern ngx_module_t ngx_http_omni_proxy_module;

/* ---- token authentication ---- */

static ngx_uint_t
omni_reload_token_ok(ngx_http_request_t *r, ngx_http_omni_loc_conf_t *plcf)
{
    /* auth switch off (default) - allow without authentication */
    if (plcf->reload_auth_enabled != 1) {
        return 1;
    }

    /* auth enabled but token not configured - reject */
    if (plcf->reload_token.len == 0) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
            "reload_auth is on but reload_token is not configured, rejecting");
        return 0;
    }

    /* verify Bearer token */
    if (r->headers_in.authorization == NULL) {
        return 0;
    }
    ngx_str_t *auth = &r->headers_in.authorization->value;
    if (auth->len <= 7 ||
        ngx_strncasecmp(auth->data, (u_char *)"Bearer ", 7) != 0)
    {
        return 0;
    }
    ngx_str_t token = { auth->len - 7, auth->data + 7 };
    return (token.len == plcf->reload_token.len &&
            ngx_memcmp(token.data, plcf->reload_token.data, token.len) == 0);
}

/* ---- query parameter extraction ---- */

static ngx_int_t
omni_get_arg(ngx_str_t *args, const char *key, u_char *dst, size_t dst_len)
{
    size_t klen = ngx_strlen(key);
    u_char *cur = args->data, *end = args->data + args->len;

    while (cur < end) {
        u_char *amp = ngx_strlchr(cur, end, '&');
        u_char *seg_end = amp ? amp : end;
        u_char *eq = ngx_strlchr(cur, seg_end, '=');
        if (eq && (size_t)(eq - cur) == klen &&
            ngx_strncmp(cur, key, klen) == 0)
        {
            size_t vlen = (size_t)(seg_end - eq - 1);
            if (vlen >= dst_len) vlen = dst_len - 1;
            ngx_memcpy(dst, eq + 1, vlen);
            dst[vlen] = '\0';
            return (ngx_int_t)vlen;
        }
        if (!amp) break;
        cur = amp + 1;
    }
    return NGX_ERROR;
}

/* ---- endpoints format validation ---- */

static ngx_uint_t
omni_validate_endpoints(u_char *data, size_t len)
{
    if (len == 0 || len > 4096) return 0;
    for (size_t i = 0; i < len; i++) {
        u_char c = data[i];
        if (!((c >= '0' && c <= '9') || c == '.' || c == ':' || c == ',')) {
            return 0;
        }
    }
    return 1;
}

/* ---- concurrency lock ---- */

static int omni_reload_try_lock(void)
{
    int fd = open(OMNI_RELOAD_LOCK_FILE, O_CREAT | O_RDWR, 0644);
    if (fd < 0) return -1;
    if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void omni_reload_unlock(int fd)
{
    if (fd >= 0) { flock(fd, LOCK_UN); close(fd); }
}

/* ---- JSON response helper ---- */

static ngx_int_t
omni_reload_respond(ngx_http_request_t *r, ngx_uint_t status,
                    u_char *json, size_t json_len)
{
    r->headers_out.status = status;
    ngx_str_set(&r->headers_out.content_type, "application/json");
    r->headers_out.content_length_n = (off_t)json_len;

    ngx_int_t rc = ngx_http_send_header(r);
    if (rc == NGX_ERROR || rc > NGX_OK || r->header_only) return rc;

    ngx_buf_t *b = ngx_pcalloc(r->pool, sizeof(ngx_buf_t));
    if (b == NULL) return NGX_HTTP_INTERNAL_SERVER_ERROR;
    b->pos = json;
    b->last = json + json_len;
    b->memory = 1;
    b->last_buf = 1;
    ngx_chain_t out = { b, NULL };
    return ngx_http_output_filter(r, &out);
}

/* ---- get nginx conf path from ngx_cycle ---- */

static ngx_int_t
omni_get_conf_path(char *dst, size_t dst_len)
{
    extern volatile ngx_cycle_t *ngx_cycle;

    if (ngx_cycle == NULL || ngx_cycle->conf_file.len == 0) {
        return NGX_ERROR;
    }

    ngx_str_t *cf = &((ngx_cycle_t *)ngx_cycle)->conf_file;
    ngx_str_t *prefix = &((ngx_cycle_t *)ngx_cycle)->conf_prefix;

    if (cf->len > 0 && cf->data[0] == '/') {
        if (cf->len >= dst_len) return NGX_ERROR;
        ngx_memcpy(dst, cf->data, cf->len);
        dst[cf->len] = '\0';
    } else {
        size_t total = prefix->len + cf->len;
        if (total >= dst_len) return NGX_ERROR;
        ngx_memcpy(dst, prefix->data, prefix->len);
        ngx_memcpy(dst + prefix->len, cf->data, cf->len);
        dst[total] = '\0';
    }
    return NGX_OK;
}

/* ---- generate upstream block ---- */

static size_t
omni_gen_upstream(char *dst, size_t dst_len, const char *name,
                  u_char *endpoints, size_t ep_len)
{
    char *p = dst;
    char *end = dst + dst_len;
    int n;

    n = snprintf(p, (size_t)(end - p),
        "    upstream %s {\n"
        "        keepalive 2048;\n"
        "        keepalive_timeout 110s;\n"
        "        keepalive_requests 20000;\n", name);
    if (n < 0 || p + n >= end) return 0;
    p += n;

    u_char *cur = endpoints;
    u_char *ep_end = endpoints + ep_len;
    while (cur < ep_end) {
        u_char *comma = cur;
        while (comma < ep_end && *comma != ',') comma++;
        size_t addr_len = (size_t)(comma - cur);
        if (addr_len > 0) {
            n = snprintf(p, (size_t)(end - p),
                "        server %.*s max_fails=3 fail_timeout=10s;\n",
                (int)addr_len, cur);
            if (n < 0 || p + n >= end) return 0;
            p += n;
        }
        cur = comma < ep_end ? comma + 1 : ep_end;
    }

    n = snprintf(p, (size_t)(end - p), "    }");
    if (n < 0 || p + n >= end) return 0;
    p += n;

    return (size_t)(p - dst);
}

/* ---- replace upstream block in conf ---- */

static ngx_int_t
omni_replace_upstream(char *conf, size_t conf_len, size_t conf_cap,
                      const char *name, char *new_block, size_t block_len,
                      size_t *new_conf_len)
{
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "upstream %s {", name);

    char *start = strstr(conf, pattern);
    if (start == NULL) return NGX_ERROR;

    /* walk back to find leading whitespace */
    while (start > conf && *(start - 1) == ' ') start--;

    /* find closing brace */
    char *p = strstr(start, pattern) + strlen(pattern);
    int depth = 1;
    while (*p && depth > 0) {
        if (*p == '{') depth++;
        else if (*p == '}') depth--;
        p++;
    }

    /* now [start, p) is the old upstream block */
    size_t old_len = (size_t)(p - start);
    size_t tail_len = conf_len - (size_t)(p - conf);
    size_t new_total = (size_t)(start - conf) + block_len + tail_len;

    if (new_total >= conf_cap) return NGX_ERROR;

    memmove(start + block_len, p, tail_len + 1);
    memcpy(start, new_block, block_len);
    *new_conf_len = new_total;
    return NGX_OK;
}

/* ---- main handler ---- */
ngx_int_t
ngx_http_omni_proxy_reload_handler(ngx_http_request_t *r)
{
    ngx_http_omni_loc_conf_t *plcf;
    u_char *buf;
    size_t n;

    plcf = ngx_http_get_module_loc_conf(r, ngx_http_omni_proxy_module);

    /* only enabled when reload_auth_enabled is explicitly set to 1 */
    if (plcf->reload_auth_enabled != 1) {
        return NGX_DECLINED;
    }

    /* only accept POST */
    if (r->method != NGX_HTTP_POST) {
        return NGX_HTTP_NOT_ALLOWED;
    }

    ngx_int_t rc = ngx_http_discard_request_body(r);
    if (rc != NGX_OK) return rc;

    buf = ngx_palloc(r->pool, OMNI_RELOAD_BUF_SIZE);
    if (buf == NULL) return NGX_HTTP_INTERNAL_SERVER_ERROR;

    /* 1. token authentication */
    if (!omni_reload_token_ok(r, plcf)) {
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\",\"message\":\"unauthorized\"}") - buf);
        return omni_reload_respond(r, NGX_HTTP_UNAUTHORIZED, buf, n);
    }

    /* 2. parse query parameters */
    u_char prefill_ep[4096], decode_ep[4096];
    ngx_int_t pf_len = omni_get_arg(&r->args, "prefill_endpoints",
                                     prefill_ep, sizeof(prefill_ep));
    ngx_int_t dc_len = omni_get_arg(&r->args, "decode_endpoints",
                                     decode_ep, sizeof(decode_ep));

    if (pf_len == NGX_ERROR || dc_len == NGX_ERROR) {
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"missing prefill_endpoints or decode_endpoints\"}")
            - buf);
        return omni_reload_respond(r, NGX_HTTP_BAD_REQUEST, buf, n);
    }

    /* 3. validate endpoints format */
    if (!omni_validate_endpoints(prefill_ep, (size_t)pf_len) ||
        !omni_validate_endpoints(decode_ep, (size_t)dc_len))
    {
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"invalid endpoints format, "
            "expected ip:port,ip:port\"}") - buf);
        return omni_reload_respond(r, NGX_HTTP_BAD_REQUEST, buf, n);
    }

    /* 4. acquire concurrency lock */
    int lock_fd = omni_reload_try_lock();
    if (lock_fd < 0) {
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"reload already in progress\"}") - buf);
        return omni_reload_respond(r, 409, buf, n);
    }

    /* 5. get current nginx conf path */
    char conf_path[1024];
    if (omni_get_conf_path(conf_path, sizeof(conf_path)) != NGX_OK) {
        omni_reload_unlock(lock_fd);
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"cannot determine nginx conf path\"}") - buf);
        return omni_reload_respond(r, NGX_HTTP_INTERNAL_SERVER_ERROR, buf, n);
    }

    /* 6. read current conf */
    FILE *fp = fopen(conf_path, "r");
    if (fp == NULL) {
        omni_reload_unlock(lock_fd);
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"cannot read nginx conf\"}") - buf);
        return omni_reload_respond(r, NGX_HTTP_INTERNAL_SERVER_ERROR, buf, n);
    }

    char *conf_buf = malloc(OMNI_RELOAD_CONF_MAX);
    if (conf_buf == NULL) {
        fclose(fp);
        omni_reload_unlock(lock_fd);
        return NGX_HTTP_INTERNAL_SERVER_ERROR;
    }

    size_t conf_len = fread(conf_buf, 1, OMNI_RELOAD_CONF_MAX - 1, fp);
    fclose(fp);
    conf_buf[conf_len] = '\0';

    /* 7. generate new upstream blocks */
    char prefill_block[8192], decode_block[8192];
    size_t pb_len = omni_gen_upstream(prefill_block, sizeof(prefill_block),
                                      "prefill_endpoints",
                                      prefill_ep, (size_t)pf_len);
    size_t db_len = omni_gen_upstream(decode_block, sizeof(decode_block),
                                      "decode_endpoints",
                                      decode_ep, (size_t)dc_len);

    if (pb_len == 0 || db_len == 0) {
        free(conf_buf);
        omni_reload_unlock(lock_fd);
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"failed to generate upstream blocks\"}") - buf);
        return omni_reload_respond(r, NGX_HTTP_INTERNAL_SERVER_ERROR, buf, n);
    }

    /* 8. replace upstream blocks in conf */
    size_t new_conf_len = conf_len;
    if (omni_replace_upstream(conf_buf, new_conf_len, OMNI_RELOAD_CONF_MAX,
                              "prefill_endpoints", prefill_block, pb_len,
                              &new_conf_len) != NGX_OK)
    {
        free(conf_buf);
        omni_reload_unlock(lock_fd);
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"failed to replace prefill upstream\"}") - buf);
        return omni_reload_respond(r, NGX_HTTP_INTERNAL_SERVER_ERROR, buf, n);
    }

    if (omni_replace_upstream(conf_buf, new_conf_len, OMNI_RELOAD_CONF_MAX,
                              "decode_endpoints", decode_block, db_len,
                              &new_conf_len) != NGX_OK)
    {
        free(conf_buf);
        omni_reload_unlock(lock_fd);
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"failed to replace decode upstream\"}") - buf);
        return omni_reload_respond(r, NGX_HTTP_INTERNAL_SERVER_ERROR, buf, n);
    }

    /* 9. backup old conf and write new conf */
    char bak_path[1088];
    snprintf(bak_path, sizeof(bak_path), "%s_bak", conf_path);
    rename(conf_path, bak_path);

    fp = fopen(conf_path, "w");
    if (fp == NULL) {
        rename(bak_path, conf_path);
        free(conf_buf);
        omni_reload_unlock(lock_fd);
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"cannot write new nginx conf\"}") - buf);
        return omni_reload_respond(r, NGX_HTTP_INTERNAL_SERVER_ERROR, buf, n);
    }
    fwrite(conf_buf, 1, new_conf_len, fp);
    fclose(fp);
    free(conf_buf);

    /* 10. nginx -t to validate (use /proc/self/exe to find nginx binary) */
    char nginx_bin[1024];
    ssize_t bin_len = readlink("/proc/self/exe", nginx_bin, sizeof(nginx_bin) - 1);
    if (bin_len <= 0) {
        rename(bak_path, conf_path);
        omni_reload_unlock(lock_fd);
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"cannot determine nginx binary path\"}") - buf);
        return omni_reload_respond(r, NGX_HTTP_INTERNAL_SERVER_ERROR, buf, n);
    }
    nginx_bin[bin_len] = '\0';

    char test_cmd[2304];
    char test_err_file[] = "/tmp/omni_reload_test.XXXXXX";
    int tmp_fd = mkstemp(test_err_file);
    if (tmp_fd >= 0) close(tmp_fd);

    snprintf(test_cmd, sizeof(test_cmd),
        "'%s' -t -c '%s' 2>%s", nginx_bin, conf_path, test_err_file);
    system(test_cmd);

    /* check output for success marker — system() return is unreliable
       inside nginx worker due to SIGCHLD handling */
    char errmsg[512] = "";
    int test_ok = 0;
    FILE *efp = fopen(test_err_file, "r");
    if (efp) {
        size_t elen = fread(errmsg, 1, sizeof(errmsg) - 1, efp);
        fclose(efp);
        errmsg[elen] = '\0';
        if (strstr(errmsg, "test is successful") != NULL) {
            test_ok = 1;
        }
        for (size_t i = 0; i < elen; i++) {
            if (errmsg[i] == '"') errmsg[i] = '\'';
            if (errmsg[i] == '\n') errmsg[i] = ' ';
        }
    }
    unlink(test_err_file);

    if (!test_ok) {
        ngx_log_error(NGX_LOG_ERR, r->connection->log, 0,
            "reload: nginx -t failed for %s: %s", conf_path, errmsg);

        rename(bak_path, conf_path);
        omni_reload_unlock(lock_fd);
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"nginx -t failed: %s\","
            "\"conf_path\":\"%s\"}", errmsg, conf_path) - buf);
        return omni_reload_respond(r, NGX_HTTP_INTERNAL_SERVER_ERROR, buf, n);
    }

    /* 11. nginx -s reload */
    char reload_cmd[1152];
    char reload_err_file[256];
    snprintf(reload_err_file, sizeof(reload_err_file), "/tmp/omni_reload_sig.%d", (int)ngx_getpid());
    snprintf(reload_cmd, sizeof(reload_cmd), "'%s' -s reload 2>%s", nginx_bin, reload_err_file);
    system(reload_cmd);

    /* check stderr output — empty or contains "signal process started" means success */
    int reload_ok = 1;
    char reload_errmsg[256] = "";
    FILE *rfp = fopen(reload_err_file, "r");
    if (rfp) {
        size_t rlen = fread(reload_errmsg, 1, sizeof(reload_errmsg) - 1, rfp);
        fclose(rfp);
        reload_errmsg[rlen] = '\0';
        if (rlen > 0 && strstr(reload_errmsg, "signal process started") == NULL) {
            reload_ok = 0;
        }
    }
    unlink(reload_err_file);

    if (!reload_ok) {
        rename(bak_path, conf_path);
        snprintf(reload_cmd, sizeof(reload_cmd), "'%s' -s reload 2>/dev/null", nginx_bin);
        system(reload_cmd);
        omni_reload_unlock(lock_fd);
        n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
            "{\"status\":\"error\","
            "\"message\":\"nginx -s reload failed, rolled back\"}") - buf);
        return omni_reload_respond(r, NGX_HTTP_INTERNAL_SERVER_ERROR, buf, n);
    }

    omni_reload_unlock(lock_fd);

    ngx_log_error(NGX_LOG_NOTICE, r->connection->log, 0,
        "reload success: prefill=%s decode=%s", prefill_ep, decode_ep);
    n = (size_t)(ngx_snprintf(buf, OMNI_RELOAD_BUF_SIZE,
        "{\"status\":\"ok\",\"message\":\"reload completed\"}") - buf);
    return omni_reload_respond(r, NGX_HTTP_OK, buf, n);
}
